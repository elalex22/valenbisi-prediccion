# Predicción de estaciones llenas de Valenbisi

Proyecto de portfolio: estimar la probabilidad de que una estación de Valenbisi (València) esté **llena** (sin anclajes libres para dejar la bici) **exactamente dentro de 60 minutos**.

El usuario quiere entender cada decisión: avanzar paso a paso y explicar el porqué.

## Datos (`data/raw/`, no versionados)
- `valenbici-master/`: un zip por día (`DD-MM-YYYY.zip`). Cada zip trae CSV con `;` como separador, una foto de todas las estaciones cada 15 min. Periodo: 2022-12-01 a 2025-09-24. Hora Europe/Madrid.
  - **Cambio de esquema el 2023-05-22.** Antes: `number_, open, available, free, total, updated_at`. Después: `Numero, Activo, Bicis_disponibles, Espacios_libres, Espacios_totales, fecha_actualizacion`.
  - La fecha y hora de cada foto salen del nombre del CSV (`valenbici_DD-MM-YYYY_HH-MM-SS.csv`).
- `valenbici-git/`: clon superficial (`git clone --depth 1`) de `github.com/ceferra/valenbici`, el repositorio de origen, que sigue actualizándose. No tiene licencia: se usa para analizar, pero los datos no se redistribuyen.
  - Comprobado con `scripts/comprobar_clon.py`, que solo mira nombres, hashes, cabeceras y número de líneas:
    - los 951 zips anteriores al 24-09-2025 y el propio 24-09-2025 son idénticos (SHA-256) a los de `valenbici-master`;
    - los nuevos tienen la misma cabecera que después del 22-05-2023.
  - **No hay ninguna foto del 25-09-2025 al 28-06-2026** (277 días). Los datos se reanudan el 29-06-2026 y están casi completos hasta el 28-10-2026.
  - **En desarrollo solo se lee `valenbici-master`.** El clon solo lo lee el script final de la v2, filtrando por fecha.
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
    - Siguen en el Parquet (`FEATURES` en `features.py` son las 20 candidatas), pero **el modelo oficial no las usa** (ver la ablación).
  - Un test de fuga cambia y borra las fotos posteriores a t y comprueba que las features de t no cambian.
- **Festivos** (`src/valenbisi/calendario.py`): tabla escrita a mano con los nacionales, autonómicos y locales de València (2022-12 a 2025-12), revisada.
  - El 26-12-2022 no es festivo: el decreto autonómico lo cambió por el 24-06.
  - El 24-06 «recuperable» de 2024 y 2025 cuenta como festivo, porque no se trabaja.
  - El 24-04-2023 (traslado de San Vicente Mártir) cuenta como festivo, pero **no está confirmado en fuente oficial**: solo sale en prensa y en calendarios de terceros.
- **Modelo oficial: LightGBM con 15 features** (`src/valenbisi/modelo.py`, `scripts/entrenar_lightgbm.py`). Las cifras son del modelo de 15 features; entre paréntesis, las del primer modelo de 20 cuando cambian:
  - Pérdida logística **sin pesos de clase**: nada de `is_unbalance`, `scale_pos_weight` ni submuestreo de negativos, que estropearían la calibración. Si algún día no cabe en memoria, solo vale una muestra uniforme de todas las filas, y hay que avisar antes.
  - Parámetros: `num_leaves=63`, `learning_rate=0.05`, `min_data_in_leaf=1000`, `feature_fraction` y `bagging_fraction` de 0,8, y `lambda_l2=1`. No hay búsqueda de hiperparámetros todavía.
  - Se entrena solo con entrenamiento. La parada temprana usa la logloss de validación (paciencia de 100 árboles), y se queda con **1020 árboles** (1040 con 20 features). Por eso la cifra de validación sale algo optimista.
  - Cada conjunto se lee del Parquet con su propio filtro (`limites()` en `evaluacion.py`). La prueba no se ha cargado nunca.
  - **Resultado en validación**: AUC-PR **0,716** y Brier skill **0,491** (0,715 y 0,490). Logloss 0,09129.
    - Llena ahora: 0,860 y 0,675.
    - No llena ahora: 0,329 y 0,242.
  - Comprobación de que no hay fuga: una tabla de probabilidad solo con `libres` y sin machine learning ya saca AUC-PR 0,512. Con 1 anclaje libre, el 23 % de las estaciones está llena a t+60. La mejora sale de ver cuántos huecos quedan, no solo si está llena o no.
  - Calibración buena en general, pero **se queda algo corta en la zona media**: predice un 3,7 % donde la realidad es un 4,6 %, y un 10,7 % donde es un 12,2 %.
  - **Decisión: no se recalibra.** El desfase viene del periodo, no del modelo (`scripts/diagnostico_calibracion.py`, sin reentrenar):
    - La tabla por libres, que no tiene nada que sobreajustar, se queda igual de corta. Con el mismo número de libres, validación se llena más que entrenamiento: con 3 libres, 7,9 % frente a 6,5 %; con 4, 5,0 % frente a 3,8 %.
    - Ajustada sin el bloque jun–ago 2024, en ese bloque solo se queda corta un 4–10 % (real ÷ predicho de 1,04 a 1,10). En validación, un 4–45 % (1,04 con 1 libre; 1,18 con 2–3; 1,33 con 4–6; 1,45 con 7 o más). Un calibrador ajustado con ese bloque no corregiría el desfase de validación.
  - **Límite conocido: variabilidad mensual.** Observado ÷ esperado por mes (filas con al menos 1 libre, tabla por libres de todo entrenamiento):

    | Mes | 2022–23 | 2023–24 | 2024–25 |
    |---|---|---|---|
    | Sep | — | 1,199 | 1,194 (val.) |
    | Oct | — | 1,216 | 1,205 (val.) |
    | Nov | — | 1,212 | 1,288 (val.) |
    | Dic | 0,942 | 0,944 | 0,893 (val.) |
    | Ene | 0,852 | 0,884 | 0,980 (val.) |
    | Feb | 0,878 | sin datos | 1,304 (val.) |
    | Mar | 1,119 | sin datos | (prueba) |
    | Abr | 0,936 | sin datos | (prueba) |
    | May | 1,014 | sin datos | (prueba) |
    | Jun | 1,058 | 1,116 | (prueba) |
    | Jul | 0,875 | 1,013 | (prueba) |
    | Ago | 0,760 | 0,959 | (prueba) |

    - Va de 0,76 a 1,30. Parte se repite cada año (sep–nov alto, dic–ene bajo) y parte no (febrero: 0,88 en 2023 y 1,30 en 2025).
    - Ningún modelo con estas features puede prever que un mes concreto sea atípico. La calibración de cada mes puede desviarse en esa medida.
  - **Frente a la tabla por libres** (`scripts/comparar_validacion.py`, sin reentrenar): LightGBM gana en todos los tramos de `libres`. AUC-PR con 1 libre 0,417 frente a 0,243; con 2–3, 0,286 frente a 0,126; con 4–6, 0,189 frente a 0,043; con 7 o más, 0,153 frente a 0,009.
  - Importancia por permutación (caída de AUC-PR): estado actual 0,64 (`ocupacion` 0,39, `libres` 0,21, `llena` 0,15), calendario 0,19 (`minuto_del_dia`), estación 0,09 y tendencia 0,005.
    - Dentro de los subgrupos difíciles la tendencia pesa algo más: 0,011 con 1–5 libres y 0,015 en «no llena ahora». Aun así, queda lejos del estado actual, el calendario y la estación (0,12–0,28).
    - En el modelo de 20 features, el tiempo de ayer daba 0,003 en total y 0,006 en los subgrupos difíciles.
  - Figuras en `reports/figures/*_lightgbm_validacion.png` y `*_comparacion_validacion.png`, resumen en `data/processed/resumen_lightgbm.json` y `resumen_comparacion_validacion.json`, y modelo en `models/lightgbm.txt` (sin versionar).
- **Ablación** (`scripts/ablacion_lightgbm.py` → `data/processed/resumen_ablacion.json`): mismos parámetros y parada temprana, medida en validación.

  | Variante | Features | Árboles | Logloss | AUC-PR | Skill | AUC-PR no llena |
  |---|---|---|---|---|---|---|
  | Completo, semilla 42 | 20 | 1040 | 0,09149 | 0,7150 | 0,4896 | 0,3272 |
  | Completo, semilla 43 | 20 | 1043 | 0,09150 | 0,7152 | 0,4898 | 0,3273 |
  | Sin tendencia | 17 | 1016 | 0,09199 | 0,7135 | 0,4879 | 0,3206 |
  | Sin tiempo de ayer | 15 | 1020 | 0,09129 | 0,7160 | 0,4906 | 0,3288 |
  | Sin las dos cosas | 12 | 868 | 0,09168 | 0,7147 | 0,4892 | 0,3230 |

  - El ruido de semilla es de 0,00001 de logloss y 0,0002 de AUC-PR. La semilla 42 reproduce exactamente el modelo de 20 features.
  - **Se mantiene la tendencia.** Quitarla empeora la logloss en 0,0005, unas 50 veces el ruido, y empeora todos los tramos de pocos huecos.
  - **Se quita el tiempo de ayer.** Sin él mejora todo: la logloss baja 0,0002, unas 20 veces el ruido. Los árboles lo usaban para reconocer días concretos de entrenamiento. Además, quitarlo evita depender de AEMET para predecir.
  - **Modelo oficial: 15 features** (`FEATURES_MODELO` y `GRUPOS_MODELO` en `modelo.py`). Las 20 candidatas siguen en el Parquet; el modelo anterior queda en `models/lightgbm_20_features.txt` y su resumen en `data/processed/resumen_lightgbm_20_features.json`. Al reentrenarlo, el modelo oficial reproduce exactamente la variante «sin tiempo de ayer».

- **Evaluación final en la prueba: decisiones fijadas antes de mirarla** (aprobadas el 2026-10-29, con la prueba sin cargar nunca):
  - **Resultado principal**: LightGBM con las 15 features de `FEATURES_MODELO` y los mismos `PARAMETROS`, **reentrenado con entrenamiento + validación**, sin parada temprana y con un **número fijo de árboles igual al `best_iteration` del modelo de 15 features** entrenado con entrenamiento: **1020**. Motivo: el periodo importa (variabilidad mensual) y validación (sep 2024–feb 2025) son los datos más cercanos a la prueba.
    - Las cuatro referencias (constante, persistencia, frecuencia y tabla por libres) se ajustan también con entrenamiento + validación. La constante del Brier skill sale de esos mismos datos.
  - **Resultado secundario**: el modelo oficial entrenado solo con entrenamiento (`models/lightgbm.txt`), con las referencias ajustadas solo con entrenamiento. Sirve para ver cuánto aporta reentrenar con datos recientes.
  - Después de ver la prueba no se cambia ninguna decisión de modelado. Si el script tiene un error de código, se corrige, se vuelve a ejecutar y se documenta.
- **Resultado final en la prueba** (`scripts/evaluacion_final.py`, ejecutado **una vez** el 2026-10-29 desde el commit `6bf4438` con el árbol de trabajo limpio → `data/processed/resumen_prueba.json`, `models/lightgbm_final.txt`, `reports/figures/*_prueba.png`):
  - Datos: 5.262.368 filas (mar–sep 2025, 207 días); un 5,90 % están llenas a t+60, frente al 5,74 % de entrenamiento + validación. No hay estaciones nuevas.
  - Comprobación previa correcta: la persistencia y la frecuencia ajustadas solo con entrenamiento reproducen las cifras publicadas (0,3798 / 0,3255 y 0,2838 / 0,1332).
  - **Escenario principal** (entrenamiento + validación, 1020 árboles), con intervalos al 95 % por bootstrap de días (200 réplicas):

    | Modelo | AUC-PR | Brier skill | AUC-PR no llena |
    |---|---|---|---|
    | Persistencia | 0,380 | 0,326 | 0,025 |
    | Frecuencia histórica | 0,290 | 0,139 | 0,104 |
    | Tabla por libres | 0,499 [0,491–0,505] | 0,378 [0,371–0,384] | 0,186 |
    | **LightGBM** | **0,699 [0,692–0,707]** | **0,476 [0,470–0,484]** | **0,340** |

    - LightGBM frente a la tabla por libres: +0,200 de AUC-PR [0,194–0,205] y +0,098 de Brier skill [0,093–0,103]. Gana en todos los tramos de libres: con 1 libre, 0,431 frente a 0,245; con 2–3, 0,305 frente a 0,125; con 4–6, 0,207 frente a 0,043; con 7 o más, 0,144 frente a 0,010.
    - Llena ahora: 0,840 / 0,657. No llena ahora: 0,340 / 0,242.
  - **Escenario secundario** (solo entrenamiento): LightGBM 0,695 / 0,472. Reentrenar con validación aporta +0,004 de AUC-PR [0,0035–0,0046] y +0,004 de Brier skill [0,0037–0,0046]: poco, pero siempre a favor.
  - Frente a validación (0,716 / 0,491), la prueba baja un poco (0,699 / 0,476). Es lo esperable: validación se usó para la parada temprana y para elegir, y es otra época del año.
  - **Calibración**: se queda corta, más que en validación, porque la prueba se llena más. Real ÷ predicho por mes, entre 1,01 y 1,07 (media 1,04). Por tramos, predice un 4,0 % donde la realidad es un 5,4 % y un 1,4 % donde es un 2,1 %. En la zona alta acierta (0,69 frente a 0,68). Encaja con el límite conocido de variabilidad entre periodos.

## Versión 2 (en desarrollo)
- **Reparto de la v2**: entrenamiento de dic-2022 a feb-2025, validación de mar a sep 2025 (la antigua prueba) y prueba v2 con los datos nuevos. Se mantiene el hueco de 60 min en cada frontera.
- **Prueba v2: decisiones fijadas antes de mirarla** (aprobadas el 2026-10-29; la prueba v2 no se ha leído nunca):
  - Periodo: `slot_utc` desde el **2025-09-25 00:00 hora local** hasta el **28-10-2026, último día completo** (hasta medianoche local del 29-10-2026). En la práctica, los datos van del 29-06-2026 al 28-10-2026.
  - **Resultado principal**: con todas las filas de la prueba v2.
  - **Resultado secundario**: excluyendo la primera semana (29-06 a 05-07-2026, hora local), en la que faltan los retardos de 24 h y 7 días porque no hay fotos anteriores.
  - Las estaciones de la prueba v2 que no estaban en entrenamiento + validación se cuentan y se miden aparte.
  - **La prueba v2 empieza 9 meses después de los últimos datos de entrenamiento.** Mide también la deriva, así que es esperable que baje respecto a la v1. Una bajada no implica por sí sola que la v2 sea peor que la v1: para eso se compara con el LightGBM v1 reentrenado con los mismos datos.
  - El resto (modelo, variables, parámetros y referencias) se fijará en «Decisiones v2 congeladas», con la etiqueta git `v2-congelado`, antes de leer la prueba v2.

- **Meteorología horaria (candidata para la v2)** (`src/valenbisi/open_meteo.py`, `scripts/descargar_open_meteo.py` → `data/raw/open_meteo/`): Open-Meteo Historical Weather API, «best match» (ERA5, ERA5-Land e IFS), en el punto 39,47 N; −0,37 E, en UTC.
  - Licencia CC BY 4.0 («Weather data by Open-Meteo.com»). La API gratuita es solo para uso no comercial.
  - Una sola descarga, del 30-11-2022 al 28-10-2026: 34.296 horas, sin huecos.
  - **Features en t** (con `h = t` truncada a la hora): `temp`, `humedad`, `viento` y `nubosidad` instantáneos en h; `prec_1h`, la lluvia de la hora anterior a h; `prec_3h`, la suma de h−2 a h (`NaN` si falta alguna); `llueve_1h`, 1 si `prec_1h` ≥ 0,1 mm, como alternativa binaria por si la cantidad es demasiado ruidosa (la ablación decide entre las dos). No usan nada posterior a t, y hay un test que lo comprueba.
  - **Contraste con AEMET 8416** (diario, periodo de desarrollo):
    - Temperatura fiable: correlación de 0,98–0,99, con 1 °C menos de media.
    - Lluvia ruidosa: correlación de Spearman de 0,57. Detecta el 71 % de los días con ≥1 mm de AEMET, y la mitad de sus días de lluvia no los confirma AEMET.
    - El caso extremo es la DANA del 29-10-2024: 105 mm frente a 1,3 mm. La celda de ~25 km incluye la zona donde llovió.
  - **Supuesto optimista:** el reanálisis se calcula días después de los hechos; en producción habría que usar observaciones en tiempo real.

- **Features candidatas de la v2** (`src/valenbisi/features_v2.py`, `vecinas.py`, `open_meteo.py`; `scripts/construir_features_v2.py` → `data/processed/valenbisi_features_v2.parquet`, 20.422.097 filas, 110 s, pico de 2,2 GB). Son 35: las 20 de la v1, cuyas columnas son idénticas a las del Parquet de la v1, más 15 nuevas:
  - **Meteorología horaria** (7): `temp`, `humedad`, `viento`, `nubosidad`, `prec_1h`, `prec_3h` y `llueve_1h`. Sin `NaN`.
  - **Vecinas** (4): las k=5 más cercanas por haversine, fijas, sobre la foto exacta de t. `vec_ocupacion`, `vec_frac_llenas`, `vec_libres_min` y `vec_delta_libres_60` (media de libres(t) − libres(t−60)). Las vecinas sin foto se ignoran. `NaN`: 0,007 % (1,1 % la delta).
  - **Retardos alineados con la hora del objetivo** (4): `libres_obj_24h` = libres(t+60−24 h) = libres(t−23 h); `libres_obj_7d` = libres(t+60−7 d); y `cambio_24h` / `cambio_7d`, el cambio de libres de t−24 h a t+60−24 h (y lo mismo hace 7 días). Se calculan en UTC.
    - `NaN`: 7,8 % / 8,8 % (24 h) y 9,5 % / 10,5 % (7 días). Siguen a los huecos de datos: 78 % en ago-2024 a 7 días, 41 % en jun-2024 (tras el hueco de feb–may) y 26 % en dic-2022 (sin datos anteriores). En la validación v2 (mar–sep 2025) son del 0,3–4 %.
    - Aviso para la ablación: el `NaN` es más frecuente en entrenamiento que en validación, y se concentra cerca de los periodos congelados.
  - Test de fuga sobre las 35: cambiar o borrar fotos de cualquier estación posteriores a t, o la meteorología posterior, no altera las features de t.

- **Ablación v2: reglas fijadas antes de ejecutar** (aprobadas el 2026-10-29; `scripts/ablacion_v2.py`):
  - Reparto v2 (`FRONTERAS_V2` en `evaluacion.py`): entrenamiento de dic-2022 a feb-2025 y validación de mar a sep 2025, con el hueco de 60 min.
  - LightGBM con los `PARAMETROS` de la v1 y parada temprana por la logloss de validación. El Dataset se construye por bloques (`SecuenciaParquet` en `modelo.py`), que da exactamente el mismo modelo que la matriz completa (test).
  - **Etapa 1**, sobre la base (las 15 features de la v1, sin el tiempo de ayer):
    - A0 (semilla 42) y A0b (semilla 43), para medir el ruido;
    - A1 + meteorología con cantidad (`temp`, `humedad`, `viento`, `nubosidad`, `prec_1h`, `prec_3h`);
    - A2 + meteorología binaria (lo mismo con `llueve_1h` en vez de `prec_*`);
    - A3 + vecinas;
    - A4 + retardos largos.
  - **Métrica**: la logloss de validación. Ruido = |A0 − A0b|.
  - **Un grupo entra** si mejora a A0, la mejora es más del doble del ruido **y** el intervalo al 95 % de la diferencia pareada excluye el 0 (bootstrap por días, 200 réplicas).
  - **Retardos, además**: la mejora de A4 tiene que mantenerse en las filas de validación con los cuatro retardos presentes, con un intervalo que excluya el 0.
  - **Lluvia**: si entran A1 y A2, gana la cantidad solo si su intervalo frente a la binaria excluye el 0; si no, la binaria.
  - **Etapa 2** (si entran dos o más grupos):
    - A5 = la base con todos los grupos aceptados, y A5b, lo mismo con la semilla 43;
    - A5 sin cada grupo: un grupo se queda solo si quitarlo empeora, con un intervalo que excluya el 0.
  - Si no entra ningún grupo, la v2 usa las features de la v1.
  - **Medidas aparte en validación**: todas las filas, con los 4 retardos presentes, con algún retardo `NaN`, no llena ahora y libres 1–6.
  - **Simulación de la primera semana de la prueba v2**: con los modelos que tienen retardos, se vuelve a predecir toda la validación con los retardos de 7 días vacíos (días 2 a 7) y con todos los retardos vacíos (primer día), y se compara con A0. No decide nada: se documenta.
  - Salidas: `data/processed/resumen_ablacion_v2.json` y las predicciones de validación en `data/processed/ablacion_v2/*.npy`.
  - Los hiperparámetros se buscan después, con las features ya elegidas.

- **Resultado de la ablación v2** (`data/processed/resumen_ablacion_v2.json`). Se ejecutó en tres tandas:
  - La etapa 1 desde el commit `96718a0`. La ejecución se cortó durante A5 porque Claude Code la mató por falta de memoria del equipo.
  - La etapa 2 con `--reanudar` (commit `751febd`).
  - A5 otra vez, desde `550adcf`. La simulación de la primera semana de A5 con todos los retardos vaciados se había sobrescrito con la variante «A5 sin retardos», que tenía el mismo nombre.
    - Ahora las simulaciones se llaman `{variante}_sim_{escenario}`, y registrar dos veces un nombre es un error.
    - El A5 reentrenado da predicciones **idénticas** a las del original (diferencia 0), y las decisiones y las comparaciones de la etapa 2 no cambian.
    - El modelo queda en `models/ablacion_v2/A5.txt`.
  - Validación v2: 5.262.368 filas, de ellas 226.767 con algún retardo `NaN`. Entrenamiento: 15.158.645.
  - **Etapa 1** (logloss de validación; diferencia frente a A0 con intervalo al 95 %):

    | Variante | Features | Árboles | Logloss | Diferencia frente a A0 | AUC-PR | Skill | AUC-PR no llena | Logloss libres 1–6 |
    |---|---|---|---|---|---|---|---|---|
    | A0 base, semilla 42 | 15 | 1510 | 0,096428 | — | 0,6992 | 0,4764 | 0,3401 | 0,24217 |
    | A0b base, semilla 43 | 15 | 1442 | 0,096441 | ruido 0,000013 | 0,6990 | 0,4764 | 0,3401 | 0,24213 |
    | A1 + meteo con cantidad | 21 | 1523 | 0,096608 | +0,000180 [+0,000008, +0,000292] | 0,6989 | 0,4762 | 0,3408 | 0,24259 |
    | A2 + meteo binaria | 20 | 1381 | 0,096573 | +0,000145 [−0,000020, +0,000259] | 0,6991 | 0,4763 | 0,3406 | 0,24256 |
    | A3 + vecinas | 19 | 1489 | 0,095178 | **−0,001250** [−0,001429, −0,001103] | 0,7047 | 0,4823 | 0,3574 | 0,23915 |
    | A4 + retardos | 19 | 1450 | 0,095290 | **−0,001139** [−0,001333, −0,000943] | 0,7066 | 0,4823 | 0,3449 | 0,24063 |

    - **Entran las vecinas y los retardos.** Cada grupo mejora unas 90 veces el ruido, con el intervalo lejos del 0.
    - Los retardos cumplen la condición extra: en las filas con los cuatro retardos, A4 − A0 = −0,001183 [−0,001349, −0,000996]. En las filas con algún `NaN` no se nota: −0,000159 [−0,000871, +0,000672].
    - **La meteorología queda fuera en sus dos versiones.**
      - Con cantidad empeora, y el intervalo excluye el 0. La binaria también empeora, aunque su intervalo cruza el 0.
      - Tampoco mejora la logloss en «no llena ahora» ni en libres 1–6. La AUC-PR solo varía en milésimas.
      - La explicación más probable es la de la ablación v1 con el tiempo de ayer: el tiempo apenas cambia el uso a 60 min vista, una vez que el modelo ve el estado, la hora y la estación, y los árboles lo usan para reconocer días concretos de entrenamiento. Además, la lluvia de reanálisis en un punto es ruidosa (ver el contraste con AEMET).
      - Quitarla evita también depender de Open-Meteo para predecir.
  - **Etapa 2** (las vecinas y los retardos juntos):

    | Variante | Features | Árboles | Logloss | Diferencia frente a A5 | AUC-PR | Skill | AUC-PR no llena |
    |---|---|---|---|---|---|---|---|
    | A5, semilla 42 | 23 | 1557 | 0,094188 | — | 0,7111 | 0,4873 | 0,3604 |
    | A5b, semilla 43 | 23 | 1754 | 0,094124 | ruido 0,000064 | 0,7118 | 0,4877 | 0,3605 |
    | A5 sin vecinas (= A4) | 19 | 1450 | 0,095290 | +0,001102 [+0,000976, +0,001257] | 0,7066 | 0,4823 | 0,3449 |
    | A5 sin retardos (= A3) | 19 | 1489 | 0,095178 | +0,000990 [+0,000804, +0,001137] | 0,7047 | 0,4823 | 0,3574 |

    - **Se quedan los dos grupos:** quitar cualquiera empeora, con el intervalo lejos del 0. Lo que aportan se suma casi entero: A5 mejora a A0 en 0,00224, frente a 0,00125 + 0,00114 por separado.
    - El ruido de semilla con 23 features (0,000064) es unas 5 veces el de la base, pero cada grupo aporta unas 15 veces ese ruido.
    - Frente a A0, A5 sube la AUC-PR de 0,699 a 0,711 y el Brier skill de 0,476 a 0,487. En «no llena ahora», la AUC-PR pasa de 0,340 a 0,360, y en libres 1–6 la logloss baja de 0,2422 a 0,2378.
  - **Features finales de la v2 (23):** las 15 de la v1, las 4 de vecinas (`vec_ocupacion`, `vec_frac_llenas`, `vec_libres_min`, `vec_delta_libres_60`) y los 4 retardos (`libres_obj_24h`, `cambio_24h`, `libres_obj_7d`, `cambio_7d`), en el orden de `FEATURES_V2` (`decisiones.features_finales` del resumen).
  - **Simulación de la primera semana de la prueba v2** (toda la validación con los retardos vaciados; no decide nada):

    | Modelo | Escenario | Logloss | Diferencia frente a A0 | AUC-PR |
    |---|---|---|---|---|
    | A4 | días 2 a 7 (sin los retardos de 7 días) | 0,095876 | −0,000553 [−0,000748, −0,000368] | 0,7034 |
    | A4 | primer día (sin ningún retardo) | 0,097117 | +0,000688 [+0,000539, +0,000827] | 0,6959 |
    | A5 | días 2 a 7 (sin los retardos de 7 días) | 0,094758 | −0,001671 [−0,001928, −0,001435] | 0,7079 |
    | A5 | primer día (sin ningún retardo) | 0,095963 | −0,000465 [−0,000706, −0,000264] | 0,7008 |

    - **Días 2 a 7:** A5 sigue mejorando a la base. Pierde 0,00057 respecto a A5 completo.
    - **Primer día:**
      - A5 sin retardos sigue siendo algo mejor que la base, gracias a las vecinas.
      - Pero es peor que un modelo que nunca tuvo retardos: 0,095963 frente a 0,095178 de A3 (+0,00079). En entrenamiento, el `NaN` de los retardos va unido a los periodos cercanos a los huecos de datos, así que el modelo no lo trata como un simple «no sé».
      - Solo afecta a un día de los 4 meses de la prueba v2.
    - Por eso se mantiene el resultado secundario fijado (sin la primera semana) y no se cambia nada.
  - Siguiente paso: la búsqueda de hiperparámetros con estas 23 features.

## Pendiente para el README
- **Límite de Open-Meteo (ejemplo de la DANA):** el 29-10-2024, el reanálisis da 105 mm de lluvia en València y la estación de AEMET 8416 solo 1,3 mm. La celda de ~25 km incluye la zona del sur donde cayó la lluvia extrema. Ilustra que la lluvia de reanálisis en un punto es ruidosa a escala de ciudad: detecta el 71 % de los días con ≥1 mm de AEMET, y la mitad de los días en que marca lluvia AEMET no registra nada. La temperatura, en cambio, es fiable (correlación de 0,98–0,99).

## Reglas de trabajo
- Windows con PowerShell; Python 3.12 en `.venv/`.
- Explicar el plan antes de cualquier cambio grande y esperar el visto bueno.
- Tests con pytest.
- Commits pequeños, cada uno con un solo propósito.
- No modificar nunca `data/raw/`.
- Memoria: el equipo tiene 15,4 GB y suelen quedar unos 4 GB libres. Los scripts grandes vigilan la RAM con `src/valenbisi/memoria.py` y se paran si quedan menos de 1,5 GB libres. Si algo no cabe, avisar antes de reducir datos.
  - Entrenar LightGBM necesita unos 3,7 GB libres al empezar (2,2 GB de pico más el margen). Con Chrome muy cargado puede no llegar.
  - Los scripts largos llaman a `evitar_suspension()` (`memoria.py`) para que el equipo no se suspenda a mitad.
- La prueba ya se usó para el resultado final (2026-10-29). No se vuelve a usar para tomar decisiones: cualquier cambio de modelo posterior necesitaría datos nuevos para evaluarse.
