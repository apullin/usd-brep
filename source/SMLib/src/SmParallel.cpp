// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/******************************************************************//**
* FILE NAME --- SmParallel.cpp
* PURPOSE: Diagnostics of kernel work on worker threads, which the
*    calling thread reports (SmParallel.h).
**********************************************************************/

#include "StdAfx.h"

#include <SmParallel.h>
#include <nurbs.h>

namespace {

// kinds of SmDiagnosticCapture::Diagnostic
enum SmDiagnosticKind
{
  SM_DK_ERROR_MESSAGE,     // smos_ErrorMessage()
  SM_DK_WRITE_BUFFER,      // smos_WriteBuffer()
  SM_DK_NLIB_ERROR,        // N_ErrSet()
  SM_DK_NLIB_ERROR_CLEAR   // N_ErrClear()
};

// capture recording the calling thread's diagnostics, NULL = report them
SM_THREAD_LOCAL SmDiagnosticCapture * t_pDiagnosticCapture = NULL;

const TCHAR * sm_GetText(const SmDiagnosticCapture::Diagnostic & crDiagnostic, int lText)
{
  return crDiagnostic.bText[lText] ? crDiagnostic.sText[lText].c_str() : NULL;
}

// makes the call that crDiagnostic recorded
void sm_Report(const SmDiagnosticCapture::Diagnostic & crDiagnostic)
{
  switch (crDiagnostic.eKind)
    {
      case SM_DK_ERROR_MESSAGE:
        smos_ErrorMessage(crDiagnostic.lNumber, sm_GetText(crDiagnostic, 0), crDiagnostic.lLine,
                          sm_GetText(crDiagnostic, 1), sm_GetText(crDiagnostic, 2), sm_GetText(crDiagnostic, 3));
        break;
      case SM_DK_WRITE_BUFFER:
        smos_WriteBuffer(sm_GetText(crDiagnostic, 0), sm_GetText(crDiagnostic, 1), sm_GetText(crDiagnostic, 2));
        break;
      case SM_DK_NLIB_ERROR:
        N_ErrSet(crDiagnostic.lNumber, crDiagnostic.pszRoutine);
        break;
      default:
        N_ErrClear();
        break;
    }
}

// records one call in rDiagnostics, or sets rbLost; never throws
SmBoolean sm_Record(int eKind, long lNumber, ULONG lLine, const TCHAR * pszRoutine, const TCHAR * const cpTexts[4],
                    std::vector<SmDiagnosticCapture::Diagnostic> & rDiagnostics, SmBoolean & rbLost)
{
  try
    {
      SmDiagnosticCapture::Diagnostic sDiagnostic;
      sDiagnostic.eKind      = eKind;
      sDiagnostic.lNumber    = lNumber;
      sDiagnostic.lLine      = lLine;
      sDiagnostic.pszRoutine = pszRoutine;
      for (int ii = 0; ii < 4; ii++)
        {
          sDiagnostic.bText[ii] = (cpTexts && cpTexts[ii]) ? TRUE : FALSE;
          if (sDiagnostic.bText[ii])
            { sDiagnostic.sText[ii] = cpTexts[ii]; }
        }
      rDiagnostics.push_back(std::move(sDiagnostic));
    }
  catch (...)
    { rbLost = TRUE; }
  return TRUE;
}

} // end anonymous namespace

/*******************************************************************//**
PURPOSE: Constructors: record the calling thread's diagnostics; the NLib
   error state starts as the calling thread's, or as that of the thread
   calling the sm_Parallel function that runs this task.

NOTES:
***********************************************************************/
SmDiagnosticCapture::SmDiagnosticCapture()
 : m_pPrevious(t_pDiagnosticCapture),
   m_lNLibError(N_ErrGetType()),
   m_bRecording(TRUE),
   m_bLost(FALSE)
{
  t_pDiagnosticCapture = this;

} // end SmDiagnosticCapture::SmDiagnosticCapture

SmDiagnosticCapture::SmDiagnosticCapture
 (const SmTaskDiagnostics & crTasks)   // in : diagnostics of the sm_Parallel call running this task
 : m_pPrevious(t_pDiagnosticCapture),
   m_lNLibError(crTasks.m_lNLibError),
   m_bRecording(TRUE),
   m_bLost(FALSE)
{
  t_pDiagnosticCapture = this;

} // end SmDiagnosticCapture::SmDiagnosticCapture

/*******************************************************************//**
PURPOSE: Destructor: stop recording, discarding the recorded diagnostics.

NOTES:
***********************************************************************/
SmDiagnosticCapture::~SmDiagnosticCapture()
{
  Stop();

} // end SmDiagnosticCapture::~SmDiagnosticCapture

/*******************************************************************//**
PURPOSE: Stop recording: the calling thread's diagnostics go to the
   capture this one replaced, or are reported.

NOTES:
***********************************************************************/
void SmDiagnosticCapture::Stop()
{
  if (m_bRecording)
    {
      SM_ASSERT(t_pDiagnosticCapture == this);
      t_pDiagnosticCapture = m_pPrevious;
      m_bRecording = FALSE;
    }

} // end SmDiagnosticCapture::Stop

/*******************************************************************//**
PURPOSE: Record a call of smos_ErrorMessage(), smos_WriteBuffer(),
   N_ErrSet() or N_ErrClear() when the calling thread records its
   diagnostics.

RETURN: TRUE = recorded, the caller must not report it,
        FALSE= the calling thread reports its diagnostics.

NOTES: Never throws.
***********************************************************************/
SmBoolean SmDiagnosticCapture::RecordErrorMessage
 (SmStatus      eError,             // in : smos_ErrorMessage() arguments
  const TCHAR * pszFileName,
  ULONG         lLineNumber,
  const TCHAR * pszMessage,
  const TCHAR * pszMessageForFile,
  const TCHAR * pszFuncName)
{
  SmDiagnosticCapture * pCapture = t_pDiagnosticCapture;
  if (pCapture == NULL)
    { return FALSE; }
  const TCHAR * const cpTexts[4] = {pszFileName, pszMessage, pszMessageForFile, pszFuncName};
  return sm_Record(SM_DK_ERROR_MESSAGE, eError, lLineNumber, NULL, cpTexts, pCapture->m_sDiagnostics, pCapture->m_bLost);

} // end SmDiagnosticCapture::RecordErrorMessage

SmBoolean SmDiagnosticCapture::RecordWriteBuffer
 (const TCHAR * pszBuff,            // in : smos_WriteBuffer() arguments
  const TCHAR * pszBuffForFile,
  const TCHAR * pszBuffForDebugLog)
{
  SmDiagnosticCapture * pCapture = t_pDiagnosticCapture;
  if (pCapture == NULL)
    { return FALSE; }
  const TCHAR * const cpTexts[4] = {pszBuff, pszBuffForFile, pszBuffForDebugLog, NULL};
  return sm_Record(SM_DK_WRITE_BUFFER, 0, 0, NULL, cpTexts, pCapture->m_sDiagnostics, pCapture->m_bLost);

} // end SmDiagnosticCapture::RecordWriteBuffer

SmBoolean SmDiagnosticCapture::RecordNLibError
 (long          lError,             // in : N_ErrSet() arguments
  const TCHAR * pszRoutine)
{
  SmDiagnosticCapture * pCapture = t_pDiagnosticCapture;
  if (pCapture == NULL)
    { return FALSE; }
  pCapture->m_lNLibError = lError;
  return sm_Record(SM_DK_NLIB_ERROR, lError, 0, pszRoutine, NULL, pCapture->m_sDiagnostics, pCapture->m_bLost);

} // end SmDiagnosticCapture::RecordNLibError

SmBoolean SmDiagnosticCapture::RecordNLibErrorClear()
{
  SmDiagnosticCapture * pCapture = t_pDiagnosticCapture;
  if (pCapture == NULL)
    { return FALSE; }
  pCapture->m_lNLibError = 0;
  return sm_Record(SM_DK_NLIB_ERROR_CLEAR, 0, 0, NULL, NULL, pCapture->m_sDiagnostics, pCapture->m_bLost);

} // end SmDiagnosticCapture::RecordNLibErrorClear

/*******************************************************************//**
PURPOSE: The NLib error number when the calling thread records its
   diagnostics.

RETURN: TRUE = rlError set, FALSE= the calling thread uses NL_ERROR.

NOTES:
***********************************************************************/
SmBoolean SmDiagnosticCapture::GetNLibError
 (long & rlError)   // out: NLib error number
{
  const SmDiagnosticCapture * cpCapture = t_pDiagnosticCapture;
  if (cpCapture == NULL)
    { return FALSE; }
  rlError = cpCapture->m_lNLibError;
  return TRUE;

} // end SmDiagnosticCapture::GetNLibError

/*******************************************************************//**
PURPOSE: Constructor, on the thread calling the sm_Parallel function.

NOTES:
***********************************************************************/
SmTaskDiagnostics::SmTaskDiagnostics()
 : m_lNLibError(N_ErrGetType()),
   m_bLost(FALSE)
{
} // end SmTaskDiagnostics::SmTaskDiagnostics

/*******************************************************************//**
PURPOSE: Keep the diagnostics of task lTask, recorded by rCapture.

NOTES: On any thread. Never throws.
***********************************************************************/
void SmTaskDiagnostics::Add
 (size_t                lTask,      // in : task index, the order of a serial run
  SmDiagnosticCapture & rCapture)   // i/o: capture of the task, emptied
{
  if (rCapture.IsEmpty())
    { return; }
  std::lock_guard<std::mutex> sLock(m_sMutex);
  if (rCapture.m_bLost)
    { m_bLost = TRUE; }
  try
    { m_sTasks.emplace_back(lTask, std::move(rCapture.m_sDiagnostics)); }
  catch (...)
    { m_bLost = TRUE; }

} // end SmTaskDiagnostics::Add

/*******************************************************************//**
PURPOSE: Report the tasks' diagnostics on the calling thread, as a
   serial run of the tasks would have: to its capture, or as the
   original calls.

NOTES: After the tasks finished.
***********************************************************************/
void SmTaskDiagnostics::Report()
{
  std::sort(m_sTasks.begin(), m_sTasks.end(), [](const Task & crA, const Task & crB) { return crA.first < crB.first; });
  for (const Task & crTask : m_sTasks)
    {
      for (const SmDiagnosticCapture::Diagnostic & crDiagnostic : crTask.second)
        { sm_Report(crDiagnostic); }
    }
  m_sTasks.clear();

  // diagnostics that could not be recorded still make the calling thread's capture non-empty
  if (m_bLost && t_pDiagnosticCapture != NULL)
    { t_pDiagnosticCapture->m_bLost = TRUE; }
  m_bLost = FALSE;

} // end SmTaskDiagnostics::Report
