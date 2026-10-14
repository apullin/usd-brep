// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/******************************************************************//**
* FILE NAME --- SmParallel.h
* PURPOSE: Run kernel work on oneTBB worker threads as the calling thread
*    would run it, isolated from the caller's TBB work.
**********************************************************************/

#ifndef __SMPARALLEL_H__
#define __SMPARALLEL_H__

#include <SmTol.h>
#include "tbb/info.h"
#include "tbb/parallel_for.h"
#include "tbb/parallel_invoke.h"
#include "tbb/task_arena.h"
#include "tbb/task_group.h"
#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

/*******************************************************************//**
PURPOSE: Number of threads one kernel operation runs its TBB work on.

NOTES: Tessellation scales to about 8 threads; beyond that the extra
   workers mostly spin. SMLIB_MAX_THREADS overrides the default of 8
   (0 = all of TBB's default concurrency).
***********************************************************************/
inline int sm_MaxThreads()
{
  static const int slMaxThreads = []()
    {
      const int lDefault = tbb::info::default_concurrency();
      const char * pszValue = std::getenv("SMLIB_MAX_THREADS");
      const int lRequested = pszValue ? std::atoi(pszValue) : 8;
      return (lRequested <= 0) ? lDefault : std::min(lRequested, lDefault);
    }();
  return slMaxThreads;
}

/*******************************************************************//**
PURPOSE: The calling thread's arena for kernel TBB work, created on first
   use: at most sm_MaxThreads() threads, or only the calling thread in
   SM_DEBUG_CODE builds, whose error and assertion reporting isn't
   thread-safe.

NOTES: NULL once the thread's thread_local objects are being destroyed,
   for kernel work started from other thread_local or static destructors.
***********************************************************************/
inline tbb::task_arena * sm_KernelArena()
{
  static thread_local bool tbDestroyed = false;  // trivially destructible: readable to the end
  struct Owner
  {
    tbb::task_arena * pArena = NULL;
    ~Owner() { tbDestroyed = true; delete pArena; }
  };
  static thread_local Owner tOwner;
  if (tbDestroyed)
    { return NULL; }
  if (tOwner.pArena == NULL)
    {
#ifdef SM_DEBUG_CODE
      tOwner.pArena = new tbb::task_arena(1, 1);
#else
      tOwner.pArena = new tbb::task_arena(sm_MaxThreads(), 1);
#endif
    }
  return tOwner.pArena;
}

/*******************************************************************//**
PURPOSE: Number of threads the sm_Parallel functions use when called from
   the calling thread now, for skipping parallel set-up that can't pay off.
***********************************************************************/
inline int sm_ParallelThreads()
{
  tbb::task_arena * pArena = sm_KernelArena();
  return pArena ? std::min(tbb::this_task_arena::max_concurrency(), pArena->max_concurrency()) : 1;
}

/*******************************************************************//**
PURPOSE: Run crFn, which starts TBB algorithms, isolated from the caller's
   TBB work, on at most sm_MaxThreads() threads. Use sm_ParallelFor() and
   sm_ParallelInvoke() rather than calling this directly.

NOTES: 1. crFn runs on the calling thread in a task group with its own
          task_group_context, which the TBB algorithms it starts bind to,
          in sm_KernelArena() unless the current arena is no wider (as
          when called from tasks already running there), or on the calling
          thread alone during thread exit. A host that
          tessellates from inside its own TBB tasks can't make them skip
          iterations by cancelling its task group, and while the calling
          thread waits here it only runs tasks that crFn spawned, never
          the host's other tasks, which may hold locks or reenter the
          kernel.
       2. The tasks run with the calling thread's floating-point settings.
       3. Exceptions thrown by crFn propagate to the caller.
***********************************************************************/
template <class Fn>
void sm_RunIsolated(const Fn & crFn)
{
  tbb::task_group_context sContext(tbb::task_group_context::isolated);
#if __TBB_FP_CONTEXT
  sContext.capture_fp_settings();
#endif
  auto Run = [&crFn, &sContext]()
    {
      tbb::task_group sGroup(sContext);
      tbb::this_task_arena::isolate([&] { sGroup.run_and_wait(crFn); });
    };
  tbb::task_arena * pArena = sm_KernelArena();
  if (pArena == NULL)
    {
      tbb::task_arena sSerial(1, 1);
      sSerial.execute(Run);
    }
  else if (tbb::this_task_arena::max_concurrency() <= pArena->max_concurrency())
    { Run(); }
  else
    { pArena->execute(Run); }
}

/*******************************************************************//**
PURPOSE: While in scope, the calling thread has the given tolerance
   defaults (SmTol thread-local state); its own come back afterwards.
***********************************************************************/
class SmThreadDefaultsScope
{
public:
  explicit SmThreadDefaultsScope(const SmTol::ThreadDefaults & crDefaults)
    : m_sPrevious(SmTol::GetThreadDefaults())
    { SmTol::SetThreadDefaults(crDefaults); }
  ~SmThreadDefaultsScope() { SmTol::SetThreadDefaults(m_sPrevious); }

  SmThreadDefaultsScope(const SmThreadDefaultsScope &) = delete;
  SmThreadDefaultsScope & operator=(const SmThreadDefaultsScope &) = delete;

private:
  SmTol::ThreadDefaults m_sPrevious;
};

class SmTaskDiagnostics;

/*******************************************************************//**
PURPOSE: While in scope, the calling thread's kernel diagnostics are
   recorded here instead of being reported: smos_ErrorMessage() and
   smos_WriteBuffer() calls, which reach host error handlers, callbacks
   and output files, and the NLib error state of N_ErrSet(), N_ErrClear()
   and N_ErrGetType(), which is otherwise the process global NL_ERROR.

NOTES: 1. Captures nest; stop or destroy them in reverse order, on the
          thread that made them. Recording never throws: a diagnostic that
          can't be recorded still makes the capture non-empty.
       2. The sm_Parallel functions run every task in a capture and then
          report the tasks' diagnostics on the calling thread, in the order
          of a serial run of the tasks: worker threads never report.
       3. Precomputations put a capture around those calls and discard
          their results when it isn't empty: the serial code then computes
          them again, reporting as a serial run.
***********************************************************************/
class SmDiagnosticCapture
{
public:
  SmDiagnosticCapture();                                         // NLib error state of the calling thread
  explicit SmDiagnosticCapture(const SmTaskDiagnostics & crTasks); // NLib error state of crTasks' calling thread
  ~SmDiagnosticCapture();

  SmDiagnosticCapture(const SmDiagnosticCapture &) = delete;
  SmDiagnosticCapture & operator=(const SmDiagnosticCapture &) = delete;

  // stops recording before the end of the scope; IsEmpty() still tells whether anything was recorded
  void      Stop();
  SmBoolean IsEmpty() const { return (m_sDiagnostics.empty() && !m_bLost) ? TRUE : FALSE; }

  // for smos_ErrorMessage(), smos_WriteBuffer(), N_ErrSet() and N_ErrClear(): TRUE when the
  // calling thread records diagnostics, which then records this call
  static SmBoolean RecordErrorMessage(SmStatus      eError,
                                      const TCHAR * pszFileName,
                                      ULONG         lLineNumber,
                                      const TCHAR * pszMessage,
                                      const TCHAR * pszMessageForFile,
                                      const TCHAR * pszFuncName);
  static SmBoolean RecordWriteBuffer(const TCHAR * pszBuff, const TCHAR * pszBuffForFile, const TCHAR * pszBuffForDebugLog);
  static SmBoolean RecordNLibError(long lError, const TCHAR * pszRoutine);
  static SmBoolean RecordNLibErrorClear();

  // for N_ErrGetType(): TRUE when the calling thread records diagnostics, with its NLib error number
  static SmBoolean GetNLibError(long & rlError);

  // one recorded call
  struct Diagnostic
  {
    int                      eKind;       // SM_DK_* in SmParallel.cpp
    long                     lNumber;     // error number or NLib error number
    ULONG                    lLine;       // line number
    const TCHAR            * pszRoutine;  // NLib routine name (static)
    SmBoolean                bText[4];    // which texts were given
    std::basic_string<TCHAR> sText[4];    // file, message, message for file, function / the three buffers
  };

private:
  friend class SmTaskDiagnostics;

  SmDiagnosticCapture   * m_pPrevious;     // capture that this one replaced on the thread
  long                    m_lNLibError;    // NLib error number while recording
  SmBoolean               m_bRecording;    // until Stop()
  SmBoolean               m_bLost;         // a diagnostic could not be recorded
  std::vector<Diagnostic> m_sDiagnostics;  // in order
};

/*******************************************************************//**
PURPOSE: Diagnostics of the tasks of one sm_Parallel call, which the
   calling thread reports afterwards in task order (SmDiagnosticCapture).
***********************************************************************/
class SmTaskDiagnostics
{
public:
  SmTaskDiagnostics();   // on the calling thread

  SmTaskDiagnostics(const SmTaskDiagnostics &) = delete;
  SmTaskDiagnostics & operator=(const SmTaskDiagnostics &) = delete;

  // on any thread: keeps the diagnostics of task lTask (tasks run serially in lTask order)
  void Add(size_t lTask, SmDiagnosticCapture & rCapture);

  // on the calling thread: reports the tasks' diagnostics as a serial run of the tasks
  void Report();

private:
  friend class SmDiagnosticCapture;
  typedef std::pair<size_t, std::vector<SmDiagnosticCapture::Diagnostic>> Task;

  long              m_lNLibError;  // of the calling thread
  std::mutex        m_sMutex;
  std::vector<Task> m_sTasks;      // tasks with diagnostics
  SmBoolean         m_bLost;       // a task's diagnostic could not be recorded
};

/*******************************************************************//**
PURPOSE: tbb::parallel_for() for kernel work: isolated and capped as in
   sm_RunIsolated(), and every chunk runs with the calling thread's
   tolerance defaults, as it would on the calling thread. The calling
   thread reports the chunks' diagnostics afterwards (SmDiagnosticCapture).
***********************************************************************/
template <class Range, class Body>
void sm_ParallelFor(const Range & crRange, const Body & crBody)
{
  const SmTol::ThreadDefaults sDefaults = SmTol::GetThreadDefaults();
  SmTaskDiagnostics sDiagnostics;
  sm_RunIsolated([&]
    {
      tbb::parallel_for(crRange, [&](const Range & crChunk)
        {
          SmThreadDefaultsScope sScope(sDefaults);
          SmDiagnosticCapture   sCapture(sDiagnostics);
          crBody(crChunk);
          sDiagnostics.Add((size_t)crChunk.begin(), sCapture);
        });
    });
  sDiagnostics.Report();
}

template <class Index, class Function>
void sm_ParallelFor(Index first, Index last, const Function & crFunction)
{
  sm_ParallelFor(tbb::blocked_range<Index>(first, last), [&](const tbb::blocked_range<Index> & crChunk)
    {
      for (Index ii = crChunk.begin(); ii != crChunk.end(); ++ii)
        { crFunction(ii); }
    });
}

/*******************************************************************//**
PURPOSE: tbb::parallel_invoke() for kernel work, as sm_ParallelFor().
***********************************************************************/
template <class F1, class F2>
void sm_ParallelInvoke(const F1 & crF1, const F2 & crF2)
{
  const SmTol::ThreadDefaults sDefaults = SmTol::GetThreadDefaults();
  SmTaskDiagnostics sDiagnostics;
  sm_RunIsolated([&]
    {
      tbb::parallel_invoke([&]
                             {
                               SmThreadDefaultsScope sScope(sDefaults);
                               SmDiagnosticCapture   sCapture(sDiagnostics);
                               crF1();
                               sDiagnostics.Add(0, sCapture);
                             },
                           [&]
                             {
                               SmThreadDefaultsScope sScope(sDefaults);
                               SmDiagnosticCapture   sCapture(sDiagnostics);
                               crF2();
                               sDiagnostics.Add(1, sCapture);
                             });
    });
  sDiagnostics.Report();
}

#endif // !__SMPARALLEL_H__
