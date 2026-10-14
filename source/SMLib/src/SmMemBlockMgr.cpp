// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**********************************************************************//**
* FILE NAME --- SmMemBlockMgr.cpp
* PURPOSE: Source file for memory block manager
**********************************************************************/

#include "StdAfx.h"

#include <SmMemBlockMgr.h>

#include <cstdint>

#include "tbb/parallel_for.h"
#include <SmParallel.h>

SmArenaMemBlockMgr::SmArenaMemBlockMgr()
{
  Reset();
}

SmArenaMemBlockMgr::~SmArenaMemBlockMgr()
{
  Reset();
}

void SmArenaMemBlockMgr::Initialize(std::size_t initialBlockSize)
{
  Reset();
  m_initialBlockSize = ALIGN_SIZE(initialBlockSize);
  Reserve(initialBlockSize);
}

std::size_t SmArenaMemBlockMgr::NewBlockSize(std::size_t bytes) const
{
  SM_ASSERT(m_initialBlockSize > 0); // Error: Initialize() has not been called

  if (m_lastBlockSize == 0) // for the first block
  {
    if (bytes <= m_initialBlockSize)
    {
      return m_initialBlockSize;
    }
    else
    {
      return ALIGN_SIZE(bytes);
    }
  }
  else
  {
    if (bytes <= m_lastBlockSize)
    {
      return 2 * m_lastBlockSize;
    }
    else
    {
      return ALIGN_SIZE(bytes);
    }
  }
}

void* SmArenaMemBlockMgr::Allocate(std::size_t bytes)
{
  Reserve(bytes);

  void* ret = (void*)&(m_blocks.GetLast()[m_nextIndexInBlock]);
  m_nextIndexInBlock += bytes;
  return ret;
}

void SmArenaMemBlockMgr::Reserve(std::size_t bytes)
{
  SM_ASSERT(m_initialBlockSize > 0); // Error: Initialize() has not been called

  if (m_nextIndexInBlock + bytes > m_lastBlockSize)
  {
    m_lastBlockSize = NewBlockSize(bytes);
    m_blocks.Add((char*) smos_Calloc(m_lastBlockSize, 1));
    m_nextIndexInBlock = 0;
    m_totalSize += m_lastBlockSize;
    SmArenaPageIn::NewBlock(this);
  }
}

std::size_t SmArenaMemBlockMgr::Size() const
{
  return m_totalSize;
}

void SmArenaMemBlockMgr::Reset()
{
  m_initialBlockSize = 0;
  m_lastBlockSize = 0;
  m_nextIndexInBlock = 0;
  m_totalSize = 0;
  for (char* p : m_blocks)
  {
    smos_Free((void*) p);
  }
  m_blocks.RemoveAll();
}

// innermost SmArenaPageIn in scope on this thread
static thread_local SmArenaPageIn * t_pArenaPageIn = NULL;

/*******************************************************************//**
PURPOSE: Write a zero into every memory page that starts in
  [pStart, pStart + lBytes), on worker threads.

NOTES: Strides by 4 KB, the smallest page size in use; larger pages are
  just written more than once. Throws nothing: the pages only save page
  faults, so if the parallel loop throws, the rest are left to the caller.
***********************************************************************/
static void sm_TouchPages
 (char        * pStart,   // in : start of memory nobody else uses meanwhile
  std::size_t   lBytes)   // in : size of that memory
{
  static constexpr std::size_t slPageStride = 4096;

  std::size_t lFirst = (slPageStride - (std::uintptr_t)pStart % slPageStride) % slPageStride;  // first page start
  if (lFirst >= lBytes)
    { return; }
  std::size_t lPages = (lBytes - lFirst + slPageStride - 1) / slPageStride;
  try
    {
      sm_ParallelFor(tbb::blocked_range<std::size_t>(0, lPages, 64), [&](const tbb::blocked_range<std::size_t> & crRange)
        {
          for (std::size_t lPage = crRange.begin(); lPage != crRange.end(); ++lPage)
            { ((volatile char *)pStart)[lFirst + lPage * slPageStride] = 0; }
        });
    }
  catch (...)
    { }
}

SmArenaPageIn::SmArenaPageIn(SmArenaMemBlockMgr & rArena, std::size_t lBytes)
  : m_cpArena(&rArena), m_lBytesLeft(lBytes), m_pPrevious(t_pArenaPageIn)
{
  // the rest of the current block
  char * pBlock = rArena.m_blocks.GetSize() > 0 ? rArena.m_blocks.GetLast() : NULL;
  if (pBlock)
    { PageIn(pBlock + rArena.m_nextIndexInBlock, rArena.m_lastBlockSize - rArena.m_nextIndexInBlock); }

  // publish only once constructed: if PageIn() throws, no destructor would unpublish
  t_pArenaPageIn = this;
}

SmArenaPageIn::~SmArenaPageIn()
{
  t_pArenaPageIn = m_pPrevious;
}

void SmArenaPageIn::NewBlock(const SmArenaMemBlockMgr * cpArena)
{
  SmArenaPageIn * pPageIn = t_pArenaPageIn;
  char          * pBlock  = cpArena->m_blocks.GetSize() > 0 ? cpArena->m_blocks.GetLast() : NULL;
  if (pPageIn && pPageIn->m_cpArena == cpArena && pBlock)  // no work for failed allocations
    { pPageIn->PageIn(pBlock, cpArena->m_lastBlockSize); }
}

void SmArenaPageIn::PageIn(char * pStart, std::size_t lSize)
{
  // leave small pieces to the calling thread
  static constexpr std::size_t slMinBytes = std::size_t(64) << 10;

  std::size_t lBytes = smos_Min(lSize, m_lBytesLeft);
  m_lBytesLeft -= lBytes;
  if (lBytes >= slMinBytes)
    { sm_TouchPages(pStart, lBytes); }
}
