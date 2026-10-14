// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/******************************************************************//**
* FILE NAME --- SmAObject.h
* PURPOSE: Header file for SmAObject.
**********************************************************************/


#ifndef __SMAOBJECT_H__
#define __SMAOBJECT_H__

#ifndef __SMOS_TYPES_H__
#include <SmTypes.h>
#endif

#ifndef __SMOBJECT_H__
#include <SmObject.h>
#endif

#ifndef __SMCACHEMGR_H__
#include <SmCacheMgr.h>
#endif

/*******************************************************************//**
PURPOSE: This is the base level for all objects which have attributes.

NOTES: 
***********************************************************************/
class SM_EXPORT SmAObject : public SmObject
{
protected:
  SmObject * m_pAttributes;  //   0  attributes = NULL
                             //   1  attribute  = pointer to an attribute or
                             // Many attributes = a pointer to an array containing attributes.
                             // type is oneof: NULL,
                             //                SmAttribute *, or
                             //                SmTArray<SmAttribute*> *
                             // test with    : if(m_pAttributes->IsKindOf(SmAttribute_TYPE))

  mutable SmCacheObj * m_pCacheObj = nullptr ;  // Cache objects used by SmCurve, SmSurface, and SmSAGObject

public:
  // constructor
  SmAObject() : m_pAttributes(NULL), 
                m_pCacheObj  (NULL) {}

  // destructor - manages m_pAttributes array and member memory
  virtual ~SmAObject()
  {
      ClearAttributes();
      SmCacheMgr::DeleteObjectsCache( this );
  }

  SmAObject(const SmAObject & crObject);

  virtual void Notify
  (
    SmNotifyOperation eNotifyOperation,   //  |       event                | caller      |  pData1  | pData2                | pData3                       
    SmObject * pData1,                    //  |----------------------------+-------------+----------+-----------------------+--------------------------    
    SmObject * pData2,                    //  | SM_NO_ADD_TO_BREP          | Brep        | AddObj   | Brep                  | AddObj's GeomPtr or NULL     
    SmObject * pData3                     //  | SM_NO_SPLIT_IN_BREP        | Brep/TopoObj| OrigObj  | Child1                | Child2                       
                                          //  | SM_NO_MERGE_IN_BREP        | Brep/TopoObj| SurvObj  | DelObj                | Brep                         
                                          //  | SM_NO_TRIM_NO_SPLIT_IN_BREP| Brep        | TgtObj   | AddedBndryObj         | NULL                     
                                          //  | SM_NO_COINCIDENT           | BrepA       | BrepAObj | BrepBObj              | BrepB                        
                                          //  | SM_NO_RM_FROM_BREP         | Brep        | RmObj    | Brep                  | RmObj's GeomPtr or NULL      
                                          //  | SM_NO_CHANGE_GEOMETRY      | TopoObj     | NewGeom  | Brep or NULL          | OldGeom or NULL              
                                          //  | SM_NO_CHANGE_OWNER         | GeomObj     | NewOwner | NewOwner Brep or NULL | OldOwner or NULL             
                                          //  | SM_NO_CONSTRUCTION         | NewObj      |  NewObj  | CopyFromObj or NULL   | NULL                         
                                          //  | SM_NO_COPY                 | FromObj     | ToObj    | ToObj's Owner or NULL | FromObj's Owner or NULL      
                                          //  | SM_NO_PRE_EDIT             | EditObj     | EditObj  | EditObj Owner or NULL | NULL                         
                                          //  | SM_NO_POST_EDIT            | EditObj     | EditObj  | EditObj Owner or NULL | NULL                         
                                          //  | SM_NO_SPLIT                | SplitGeomObj| Child1   | Child2                | SplitObj's Owner or NULL     
                                          //  | SM_NO_MERGE                | MergeGeomObj| OrigObj1 | OrigObj2              | MergeObj's Owner or NULL     
                                          //  | SM_NO_REG_PROPAGATION      | MergeReg    | ThisRegs | OtherBrep->SrcRegs    | ThisBrep->MergeReg           
                                          //  | SM_NO_DESTRUCTION          | DelObj      | DelObj   |  NULL                 |  NULL                        
  ) ;

  // add attribute to this object - call AddUser() on attribute - Don't add same attribute type more than one time
  void AddAttribute(SmAttribute * pAttribute); // ptr is stored and not copied.

  // remove attribute from this object - call RemoveUser on attribute
  void RemoveAttribute(SmAttribute * pAttribute, SmBoolean bDoNotDelete = FALSE);

  // remove all attributes from this object and set m_Attributes ptr to NULL
  void ClearAttributes() ;

  // get all of the attributes of this object
  void GetAttributes(SmTArray<SmAttribute*> & rAttributes) const;

  // return TRUE when m_pAttributes is not NULL
  SmBoolean HasAttributes(void) const    { return (m_pAttributes != NULL); }

  // find attribute by AttributeID - only one attribute of a type allowed on a given base object.
  SmAttribute * FindAttribute(ULONG lAttributeID) const; 

  // find attribute by AttributeID looking up the SmTopology Owners tree
  SmAttribute *FindInheritedAttribute(ULONG lAttributeID) const;

  // Add attribute to this object - add attribute without checking for duplicate types
  void AddSlewAttribute(SmAttribute * pAttribute);

  // retrieve the all attributes of this type
  void FindSlewOfAttributes(ULONG lAttributeID, SmTArray<SmAttribute*> & rAttributes) const;

  // return inheritance owner for this object
  virtual SmAObject *GetAOwner() const ;

  // get memory used pointed to by m_pAttributes
  ULONG GetAttributeMemoryUsed
  (
    ULONG &rlMemoryAllocated, 
     SmMarkType eMarkType                     ///< [in] : uses without increment argument mark values <br>
  ) const ;  

  // Simple cache access methods
  SmCacheObj          * GetCacheObj() const                       { return m_pCacheObj ; }
  void                  SetCacheObj(SmCacheObj * pCacheObj) const { m_pCacheObj = pCacheObj; }

  // used by SmCacheMgr::GetOrCreateObjectCache()
  // create a new or fetch an existing ObjectCache - place ObjectCache in m_pCacheObj
  virtual SmStatus CacheMakeOrValidate
  (
    SmObjectCacheType  eObjectCacheType,    ///< [in] : oneof SM_OC_CURVE                                <br>
                                            ///<              SM_OC_SURFACE                              <br>
                                            ///<              SM_OC_TRIMSRF                              <br>
                                            ///<              SM_OC_BREP                                 <br>
    const SmAObject  * pObject,             ///< [in] : target object                                    <br>
    SmCacheObj       * pOldCache,           ///< [in] : existing target Object's ObjectCache or NULL     <br>
    SmCacheObj      *& rpPNewCache          ///< [out]: ptr to target object's ObjectCache               <br>  
  ) const ;

  virtual SmBoolean AssertValid
  (
    SmAssertArray    * pAList=NULL,           ///< [in,out]: Accumulating list of failed Asserts, NULL to ignore                                            <br>
    SmAssertTestLevel  eTestLevel=SM_LEVEL_0, ///< [in] : SM_LEVEL_0=fewest tests, SM_LEVEL_1, SM_LEVEL_2=all tests,                                        <br>
                                              ///<      : SM_LEVEL_GIVEN = run tests in order requested in pTestRequests                                    <br>
    SmAssertWalking    eWalkTree=SM_WALK,     ///< NotUsed: [in] : SM_WALK = run AssertValid on any topology graph descendants, SM_NO_WALK=don't, default:[SM_WALK]  <br>
    SmTArray<ULONG>  * pTestRequests=NULL     ///< [in] : when eTestLevel == SM_LEVEL_GIVEN, run these tests in this order, default:[NULL]                  <br>
  ) const ;


  // obsolete
  //  virtual SmBoolean AssertHeal (SmAssertReport & rAReport, SmAssertArray * pAList) ;
  
  // Define  GetType(), IsKindOf(), and virtual Dump() (Dump needs local implementation)
  SM_COMMON(SmAObject,SmObject,SmAObject_TYPE);
      

}; // end class SmAObjecct

/*******************************************************************//**
PURPOSE: While in scope, the calling thread records which of the watched
   curves and surfaces are edited (SM_NO_PRE_EDIT) or destroyed, so that
   results computed from them earlier can be checked before they are used.

NOTES: Edits made on other threads are not recorded.
***********************************************************************/
class SmGeometryEditTracker
{
public:
  SmGeometryEditTracker();
  ~SmGeometryEditTracker();

  SmGeometryEditTracker(const SmGeometryEditTracker &) = delete;
  SmGeometryEditTracker & operator=(const SmGeometryEditTracker &) = delete;

  // records edits of cpObject from now on
  void Watch(const SmAObject * cpObject);

  // TRUE when cpObject is not watched or was edited or destroyed since it was watched
  SmBoolean WasEdited(const SmAObject * cpObject) const;

  // called by SmCurve::Notify() and SmSurface::Notify() for edits and destruction
  static void NoteEdit(const SmAObject * cpObject);

private:
  void * m_pEdited;    // watched objects and their edit state
  void * m_pPrevious;  // tracker that was active before this one
} ;

// called by SmCurve::CacheMakeOrValidate(): FALSE when cpOldCache, the cache of cpCurve, was built
// ahead of time on a worker thread (SmUVTrimCurvePrecompute) with a tolerance that has changed since
SmBoolean sm_CheckPrecomputedCurveCache(const SmAObject * cpCurve, const SmCacheObj * cpOldCache);

#endif // !__SMAOBJECT_H__


