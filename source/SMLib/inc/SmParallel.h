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

/*******************************************************************//**
PURPOSE: tbb::parallel_for() for kernel work: isolated and capped as in
   sm_RunIsolated(), and every chunk runs with the calling thread's
   tolerance defaults, as it would on the calling thread.
***********************************************************************/
template <class Range, class Body>
void sm_ParallelFor(const Range & crRange, const Body & crBody)
{
  const SmTol::ThreadDefaults sDefaults = SmTol::GetThreadDefaults();
  sm_RunIsolated([&]
    {
      tbb::parallel_for(crRange, [&](const Range & crChunk)
        {
          SmThreadDefaultsScope sScope(sDefaults);
          crBody(crChunk);
        });
    });
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
  sm_RunIsolated([&]
    {
      tbb::parallel_invoke([&] { SmThreadDefaultsScope sScope(sDefaults); crF1(); },
                           [&] { SmThreadDefaultsScope sScope(sDefaults); crF2(); });
    });
}

#endif // !__SMPARALLEL_H__
