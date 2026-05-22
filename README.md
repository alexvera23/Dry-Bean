Optimización Híbrida de Hiperparámetros de un MLP mediante AG-GWO

Este repositorio contiene la implementación formal correspondiente al examen del Tercer Parcial de la materia de Tecnología de Inteligencia Artificial. El proyecto propone y evalúa una arquitectura metaheurística híbrida que combina el Algoritmo Genético (AG) y el Optimización por Lobo Gris (GWO) para la sintonización automática de hiperparámetros en una Red Neuronal de Perceptrón Multicapa (MLPClassifier), resolviendo el problema de clasificación multiclase del Dry Bean Dataset de la UCI.

Descripción del Proyecto

El ajuste de hiperparámetros de un Perceptrón Multicapa (MLP) representa un problema de optimización complejo, no lineal y en un espacio de búsqueda continuo-discreto. Métodos tradicionales como Grid Search sufren de la maldición de la dimensionalidad, mientras que metaheurísticas puras de explotación (como GWO) son propensas a quedar atrapadas en óptimos locales.

Nuestra Solución: Híbrido AG-GWO

Esta propuesta combina la velocidad de convergencia y explotación jerárquica del GWO (guiada por los líderes $\alpha$, $\beta$ y $\delta$) con los operadores evolutivos de exploración del AG (Selección por Torneo, Cruce Aritmético Combinado y Mutación Gaussiana Adaptativa).

El optimizador busca una solución en el hipercubo continuo $[-1, 1]^5$, la cual se decodifica en las siguientes variables del clasificador:

Capas Ocultas: Número de neuronas en la primera y segunda capa.

Alpha ($\alpha_{reg}$): Término de penalización L2 (regularización).

Tasa de Aprendizaje Inicial ($\eta$): Rapidez del descenso de gradiente.

Función de Activación: tanh, relu o logistic.

Estructura del Repositorio

La arquitectura del proyecto está estructurada de forma modular y limpia para facilitar su distribución y evaluación:

Proyecto_IA_Parcial3/
├── data/
│   └── Dry_Bean_Dataset.xlsx       # Base de datos original descargada (obligatoria)
├── outputs/                        # Almacén de gráficas de convergencia y reportes (Auto-generado)
│   ├── 01_evolucion_fitness.png
│   ├── 02_distribucion_boxplot.png
│   ├── 03_convergencia_heatmap.png
│   ├── 04_tendencia_central.png
│   ├── 05_matriz_confusion.png
│   ├── 06_resumen_final.png
│   ├── 07_serie_iteraciones.csv    # Serie de tiempo estadística por iteración
│   ├── 08_fitness_individuos.csv   # Historial detallado de toda la población
│   └── 09_resultados_finales.csv   # Hiperparámetros y métricas de inferencia final
├── env/                            # Entorno virtual de Python (excluido de git)
├── .gitignore                      # Configuración de exclusión para Git
├── main.py                         # Script principal ejecutable con el flujo AG-GWO
├── requirements.txt                # Lista de dependencias del proyecto
└── README.md                       # Documentación del proyecto (este archivo)

Requisitos de Instalación

Sigue estos pasos detallados para recrear el entorno virtual de ejecución e instalar todas las dependencias necesarias.

1. Clonar el repositorio

git clone https://github.com/alexvera23/Dry-Bean.git



2. Configurar el Entorno Virtual (Python 3.10+)

Dependiendo de tu terminal o intérprete de comandos, utiliza el bloque correspondiente para crear y activar el entorno virtual:

En Terminales Unix Tradicionales (Bash / Zsh):

# Crear el entorno virtual
python3 -m venv env

# Activar el entorno virtual
source env/bin/activate


En Fish Shell:

# Crear el entorno virtual
python3 -m venv env

# Activar el entorno virtual
source env/bin/activate.fish


En Windows (PowerShell):

# Crear el entorno virtual
python -m venv env

# Activar el entorno virtual
.\env\Scripts\Activate.ps1


(Una vez activado el entorno, notarás el prefijo (env) al inicio de tu prompt en la terminal).

3. Instalar las dependencias del proyecto

Con el entorno virtual activado, instala las dependencias obligatorias especificadas en el archivo requirements.txt:

pip install --upgrade pip
pip install -r requirements.txt


Preparación del Dataset

Debido a que el Dry Bean Dataset es de acceso libre en el repositorio UCI Machine Learning, y para no saturar el peso de este repositorio, asegúrate de colocar el archivo de datos en la ruta adecuada:

Descarga el archivo Excel desde el portal oficial de la UCI.

Coloca el archivo con el nombre exacto Dry_Bean_Dataset.xlsx dentro de la carpeta data/ en la raíz de este proyecto.

Ejecución del Proyecto

Para iniciar el pipeline completo de carga de datos, división estratificada para balancear clases, ejecución de la metaheurística de optimización híbrida de 20 iteraciones con 15 individuos, evaluación final del clasificador sobre el conjunto de prueba no visto, y guardado de resultados, simplemente ejecuta:

python main.py


Resultados Esperados en Consola

Durante la ejecución verás un monitoreo detallado por generación del progreso del Lobo Alpha (Mejor solución), la media, la desviación estándar y el tiempo acumulado. Al finalizar, el script imprimirá en pantalla:

Los mejores hiperparámetros encontrados en el hipercubo continuo decodificados.

El reporte detallado de clasificación por clase (Precision, Recall, F1-Score, Support) para el conjunto de prueba.

La precisión (Accuracy) final global obtenida.

Análisis y Gráficas de Salida

Tras la ejecución, la carpeta outputs/ se creará de forma dinámica y contendrá un análisis visual de alto impacto para documentar el comportamiento estadístico de la optimización:

01_evolucion_fitness.png: Curva de convergencia histórica del fitness máximo y la media de la población.

02_distribucion_boxplot.png: Diagrama de caja y bigotes que visualiza cómo disminuye la variabilidad del fitness conforme progresa la evolución.

03_convergencia_heatmap.png: Mapa de calor que traza el desempeño individual de cada uno de los 15 miembros de la población a través de las 20 iteraciones.

04_tendencia_central.png: Gráfica de momentos estadísticos complementarios (Desviación estándar, asimetría, curtosis) para auditar el balance exploración/explotación.

05_matriz_confusion.png: Matriz de confusión final en el conjunto de prueba para evaluar el desbalance y la discriminación interclase.

06_resumen_final.png: Visualización de la topología final de la red y radar de hiperparámetros seleccionados.

Archivos CSV (07_serie_iteraciones.csv, etc.): Tabulación de datos crudos para análisis científico profundo.

Autores y Licencia

Integrantes: [Alejandro Cholula OLvera] & [Fatima Mentado Giron]

Licencia MIT. Uso libre con fines académicos.