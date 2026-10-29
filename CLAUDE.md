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
- **Evaluación** (`src/valenbisi/evaluacion.py`):
  - Reparto temporal por `slot_utc`, con fronteras a medianoche Europe/Madrid. Entrenamiento hasta el 2024-08-31, validación del 2024-09-01 al 2025-02-28 y prueba desde el 2025-03-01.
  - Hueco de 60 min antes de cada frontera: una fila solo entra en un conjunto si su t+60 también cae en él.
  - Todo lo que se ajusta (referencias, modelos) usa solo entrenamiento. Validación sirve para elegir; prueba solo para el resultado final.
  - Métricas: AUC-PR (el azar saca la proporción de positivos), Brier y Brier skill score frente a la constante de entrenamiento, y una curva de calibración por tramos de cuantiles.
- **Referencias a batir** (`scripts/evaluar_referencias.py`, en prueba):
  - Persistencia: 62,7 % si está llena ahora y 2,3 % si no, ajustada con entrenamiento. AUC-PR 0,380, Brier skill 0,326.
  - Frecuencia por (estación, hora local de t, laborable/fin de semana), con respaldo (hora, tipo de día) si la celda tiene menos de 30 filas. AUC-PR 0,284, Brier skill 0,133. Los festivos cuentan como laborables.
  - La persistencia gana en conjunto, pero no sabe ordenar dentro de cada grupo. Entre las estaciones que no están llenas ahora, la frecuencia ordena mucho mejor (AUC-PR 0,100 frente a 0,025). El modelo tiene que combinar las dos cosas.
  - En validación: persistencia AUC-PR 0,393 y Brier skill 0,341; frecuencia 0,295 y 0,146.
  - **Tabla por anclajes libres** (tercera referencia, `ajustar_libres` / `predecir_libres`): proporción de llena a t+60 por valor exacto de `libres` en entrenamiento, celdas de al menos 30 filas; sin celda válida se usa la celda válida inferior. Como `libres == 0` es `llena`, en «llena ahora» coincide con la persistencia.
    - En validación: AUC-PR 0,512 y Brier skill 0,392; en «no llena ahora», 0,186 y 0,166.
    - Su cifra en prueba se calcula en la evaluación final, junto con LightGBM (`evaluar_referencias.py` no la incluye todavía).
- **Features** (`src/valenbisi/features.py`, `scripts/construir_features.py` → `data/processed/valenbisi_features.parquet`):
  - Estado actual: `llena`, `libres`, `bicis`, `total` y `ocupacion = bicis / (bicis + libres)`. Se divide entre los anclajes que funcionan, así que una estación llena con anclajes averiados da 1,0.
  - Tendencia: `delta_libres_k = libres(t) − libres(t−k)` para k = 15, 30 y 60 min.
    - La foto de t−k se busca por clave exacta (estación, `slot_utc − k`) en el Parquet **limpio**, no en el de objetivo: las fotos sin etiqueta a t+60 siguen siendo un pasado válido.
    - Si falta la foto, la delta queda `NaN`. Nunca se rellena con otra foto ni con 0. Pasa en el 0,8–1,1 % de las filas.
  - Estación: `estacion` como categórica nativa de LightGBM, más `lat` y `lon`.
  - Calendario (`src/valenbisi/calendario.py`): `minuto_del_dia`, `dia_semana`, `mes` y `festivo`, en hora local a partir de `slot_utc`.
  - Meteorología (`src/valenbisi/tiempo.py`): `tmed`, `tmin`, `tmax`, `prec` y `sol` **del día local anterior**, con el sufijo `_ayer`. El dato del mismo día sería una fuga, porque incluye horas que aún no han pasado. `"Ip"` se convierte en 0,05 mm.
  - Un test de fuga cambia y borra las fotos posteriores a t y comprueba que las features de t no cambian.
- **Festivos** (`src/valenbisi/calendario.py`): tabla escrita a mano con los nacionales, autonómicos y locales de València (2022-12 a 2025-12), revisada.
  - El 26-12-2022 no es festivo: el decreto autonómico lo cambió por el 24-06.
  - El 24-06 «recuperable» de 2024 y 2025 cuenta como festivo, porque no se trabaja.
  - El 24-04-2023 (traslado de San Vicente Mártir) cuenta como festivo, pero **no está confirmado en fuente oficial**: solo sale en prensa y en calendarios de terceros.
- **Primer modelo: LightGBM** (`src/valenbisi/modelo.py`, `scripts/entrenar_lightgbm.py`):
  - Pérdida logística **sin pesos de clase**: nada de `is_unbalance`, `scale_pos_weight` ni submuestreo de negativos, que estropearían la calibración. Si algún día no cabe en memoria, solo vale una muestra uniforme de todas las filas, y hay que avisar antes.
  - Parámetros: `num_leaves=63`, `learning_rate=0.05`, `min_data_in_leaf=1000`, `feature_fraction` y `bagging_fraction` de 0,8, y `lambda_l2=1`. No hay búsqueda de hiperparámetros todavía.
  - Se entrena solo con entrenamiento. La parada temprana usa la logloss de validación (paciencia de 100 árboles), y se queda con 1040 árboles. Por eso la cifra de validación sale algo optimista.
  - Cada conjunto se lee del Parquet con su propio filtro (`limites()` en `evaluacion.py`). La prueba no se ha cargado nunca.
  - **Resultado en validación**: AUC-PR **0,715** y Brier skill **0,490**.
    - Llena ahora: 0,859 y 0,674.
    - No llena ahora: 0,327 y 0,241.
  - Comprobación de que no hay fuga: una tabla de probabilidad solo con `libres` y sin machine learning ya saca AUC-PR 0,512. Con 1 anclaje libre, el 23 % de las estaciones está llena a t+60. La mejora sale de ver cuántos huecos quedan, no solo si está llena o no.
  - Calibración buena en general, pero **se queda algo corta en la zona media**: predice un 3,7 % donde la realidad es un 4,7 %, y un 10,7 % donde es un 12,1 %. Está pendiente decidir si se recalibra (con una parte de entrenamiento, nunca con validación).
    - La tabla por libres tiene el mismo desfase: con el mismo número de libres, validación se llena más que entrenamiento (con 3 libres, 7,9 % frente a 6,5 %; con 4, 5,0 % frente a 3,8 %). Apunta a deriva entre periodos, no a un fallo del modelo.
  - **Frente a la tabla por libres** (`scripts/comparar_validacion.py`, sin reentrenar): LightGBM gana en todos los tramos de `libres`. AUC-PR con 1 libre 0,415 frente a 0,243; con 2–3, 0,286 frente a 0,126; con 4–6, 0,188 frente a 0,043.
  - Permutación por grupo dentro de los subgrupos difíciles (caída de AUC-PR): con 1–5 libres, la tendencia da 0,010 y el tiempo de ayer 0,006; en «no llena ahora», 0,014 y 0,006. Siguen siendo pequeñas frente al estado actual, el calendario y la estación (0,12–0,27).
  - Importancia por permutación (caída de AUC-PR): estado actual 0,64 (`ocupacion` y `libres`), calendario 0,19 (`minuto_del_dia`), estación 0,09. La tendencia (0,005) y el tiempo de ayer (0,003) apenas aportan.
  - Figuras en `reports/figures/*_lightgbm_validacion.png`, resumen en `data/processed/resumen_lightgbm.json` y modelo en `models/lightgbm.txt` (sin versionar).

## Reglas de trabajo
- Windows con PowerShell; Python 3.12 en `.venv/`.
- Explicar el plan antes de cualquier cambio grande y esperar el visto bueno.
- Tests con pytest.
- Commits pequeños, cada uno con un solo propósito.
- No modificar nunca `data/raw/`.
- Memoria: el equipo tiene 15,4 GB y suelen quedar unos 4 GB libres. Los scripts grandes vigilan la RAM con `src/valenbisi/memoria.py` y se paran si quedan menos de 1,5 GB libres. Si algo no cabe, avisar antes de reducir datos.
- La prueba no se toca hasta el resultado final.
