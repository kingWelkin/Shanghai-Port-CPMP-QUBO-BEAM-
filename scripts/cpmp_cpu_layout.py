"""Read-only Windows CPU-set checks for homogeneous timed study placement."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import struct


def cpu_sets():
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    query=kernel.GetSystemCpuSetInformation
    query.argtypes=[ctypes.c_void_p,wintypes.ULONG,ctypes.POINTER(wintypes.ULONG),wintypes.HANDLE,wintypes.ULONG]
    query.restype=wintypes.BOOL
    size=wintypes.ULONG();query(None,0,ctypes.byref(size),None,0)
    if not size.value:raise ctypes.WinError(ctypes.get_last_error())
    buffer=ctypes.create_string_buffer(size.value)
    if not query(buffer,len(buffer),ctypes.byref(size),None,0):raise ctypes.WinError(ctypes.get_last_error())
    rows=[];offset=0
    while offset<size.value:
        length,kind=struct.unpack_from('<II',buffer.raw,offset)
        if length<24 or offset+length>size.value:raise ValueError('Invalid CPU-set record')
        if kind==0:
            cpu_set,group,logical,core,cache,numa,efficiency,flags=struct.unpack_from('<IHBBBBBB',buffer.raw,offset+8)
            rows.append(dict(cpu_set_id=cpu_set,group=group,logical_processor=logical,
                core_index=core,efficiency_class=efficiency))
        offset+=length
    return rows


def homogeneous_layout(cpus,expected=None):
    available={r['logical_processor']:r for r in cpu_sets() if r['group']==0}
    rows=[available[cpu] for cpu in cpus]
    if len({r['core_index'] for r in rows})!=len(rows):raise ValueError('CPU slots share physical execution resources')
    if len({r['efficiency_class'] for r in rows})!=1:raise ValueError('CPU slots mix performance classes')
    if expected is not None and rows!=expected:raise ValueError('Frozen CPU topology changed')
    return rows
