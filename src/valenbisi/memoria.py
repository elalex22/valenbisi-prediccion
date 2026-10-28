"""Memoria del proceso y del equipo (Windows), para vigilar los scripts que leen millones de filas.

Usa la API de Windows con ctypes y no necesita ``psutil``. En otros sistemas
las funciones devuelven None y la comprobación no hace nada.
"""

import ctypes
import sys
from ctypes import wintypes

GB = 1024**3


class _ContadoresProceso(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
    ]


class _EstadoMemoria(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def memoria_proceso_gb() -> tuple[float, float] | None:
    """(memoria actual, pico desde que arrancó) del proceso en GB."""
    if sys.platform != "win32":
        return None
    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    # Sin argtypes, ctypes pasaría el identificador (64 bits) como un int de 32 bits.
    kernel32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ContadoresProceso), wintypes.DWORD]
    c =_ContadoresProceso(cb=ctypes.sizeof(_ContadoresProceso))
    if not kernel32.K32GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb):
        return None
    return c.WorkingSetSize / GB, c.PeakWorkingSetSize / GB


def memoria_libre_gb() -> float | None:
    """RAM física libre en el equipo, en GB."""
    if sys.platform != "win32":
        return None
    estado = _EstadoMemoria(dwLength=ctypes.sizeof(_EstadoMemoria))
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(estado)):
        return None
    return estado.ullAvailPhys / GB


def comprobar_memoria(minimo_libre_gb: float) -> None:
    """Para el script si el equipo se queda con menos de ``minimo_libre_gb`` libres.

    Es mejor parar con un mensaje claro que dejar que Windows empiece a usar el
    disco como memoria y el equipo se quede bloqueado.
    """
    libre = memoria_libre_gb()
    if libre is not None and libre < minimo_libre_gb:
        proceso = memoria_proceso_gb()
        uso = f"; este proceso usa {proceso[0]:.1f} GB" if proceso else ""
        raise MemoryError(f"Quedan {libre:.1f} GB libres (mínimo {minimo_libre_gb} GB){uso}. Se para aquí.")
