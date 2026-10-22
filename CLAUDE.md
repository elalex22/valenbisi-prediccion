# Predicción de estaciones llenas de Valenbisi

Proyecto de portfolio: estimar la probabilidad de que una estación de Valenbisi (València) esté **llena** (sin anclajes libres para dejar la bici) **exactamente dentro de 60 minutos**.

El usuario quiere entender cada decisión: avanzar paso a paso y explicar el porqué.

## Datos (`data/raw/`, no versionados)
- `valenbici-master/`: un zip por día (`DD-MM-YYYY.zip`). Cada zip trae CSV con `;` como separador, una foto de todas las estaciones cada 15 min. Periodo: 2022-12-01 a 2025-09-24. Hora Europe/Madrid.
  - **Cambio de esquema el 2023-05-22.** Antes: `number_, open, available, free, total, updated_at`. Después: `Numero, Activo, Bicis_disponibles, Espacios_libres, Espacios_totales, fecha_actualizacion`.
  - La fecha y hora de cada foto salen del nombre del CSV (`valenbici_DD-MM-YYYY_HH-MM-SS.csv`).
- `datos_tiempo/`: JSON diarios de AEMET (estación 8416 València) en latin-1. Los números vienen como texto con coma decimal y `prec="Ip"` significa precipitación inapreciable. Algunos ficheros contienen `404 datos expirados` (2020 y el segundo semestre de 2021); no afecta al periodo de las bicis.

## Decisiones aprobadas
- **Limpieza**:
  - Descartar los periodos congelados: registros con `fecha_actualizacion` de más de 24 h respecto a la foto, o fotos idénticas a la anterior. Afecta sobre todo a ene–jun 2024 y ago 2024.
  - Descartar las fotos vacías (solo cabecera).
  - Descartar la estación 299 (de pruebas).
  - Descartar los registros de estaciones inactivas (`Activo != "T"`).
  - Descartar las filas con 0 bicis y 0 anclajes libres (anclajes averiados): no cuentan como llenas.
- **"Llena"** = estación activa con `Espacios_libres == 0`.
- **Tiempo**:
  - `ts_utc` es el tiempo de referencia: ordenar, deduplicar y calcular t+60 siempre sobre él.
  - `ts` (hora local Europe/Madrid) se conserva para las features.
  - Las fotos de la hora repetida de octubre (02:00–02:59 del último domingo) se descartan: los nombres no permiten saber de qué pasada son, porque solo cambian los segundos y los zips van en orden alfabético.
  - Una hora inexistente de marzo se desplaza a las 03:00 y se cuenta en el resumen.
  - Un zip no es siempre un día: puede traer fotos de varias fechas, y los ficheros van en orden alfabético. Hay que ordenar siempre por `ts_utc`.
- **Objetivo**: probabilidad de que la estación esté llena en t + 60 min exactos. Es una clasificación binaria desbalanceada: ~6 % de positivos.
  - `slot_utc` = `ts_utc` redondeado al múltiplo de 15 min más cercano. Da error si una foto se desvía más de 5 min o si dos fotos caen en el mismo (slot, estación).
  - `llena_60` = `llena` de la misma estación en `slot_utc + 60 min` exactos. Si no existe, la fila se descarta: nunca se rellena con t+45 ni t+75.
  - De t+60 solo se guarda `llena_60`, para evitar fugas. Las features solo pueden usar datos con `slot_utc ≤ t`.
  - El reparto entrenamiento/prueba será temporal, con un hueco de 60 min en la frontera.
  - Referencia a batir: si está llena ahora, lo está a t+60 el ~61 % de las veces; si no, el ~2,4 %.

## Reglas de trabajo
- Windows con PowerShell; Python 3.12 en `.venv/`.
- Explicar el plan antes de cualquier cambio grande y esperar el visto bueno.
- Tests con pytest.
- Commits pequeños, cada uno con un solo propósito.
- No modificar nunca `data/raw/`.
