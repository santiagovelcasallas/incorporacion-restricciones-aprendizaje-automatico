# Aclaración sobre el uso de inteligencia artificial

Este proyecto es autoría del equipo (Manuela Rendón Sepulveda, Luciana Núñez Ospina, Cristian Miguel Gómez Salazar y Santiago Alberto Vélez Casallas). Usamos un asistente de IA (Claude, de Anthropic) como **herramienta de programación y apoyo**, del mismo modo que se usa un IDE, una librería o un buscador. Algunos commits incluyen la línea automática `Co-Authored-By: Claude`, que agrega la herramienta por defecto; **no implica coautoría ni cesión de derechos**.

## Qué hizo el equipo

- **La idea y el enfoque de la exposición:** mostrar que una restricción aplicada a un Transformer ya entrenado puede mejorar su desempeño en un problema médico real, y que la lección central es *qué* restringir.
- **La investigación que llevó al caso de estudio:** cuando las primeras pruebas (BiomedCLIP en PatchCamelyon) no mostraron una mejora relevante, el equipo buscó en la literatura casos donde una restricción sí ayuda en imágenes diagnósticas (TransUNet con priors espaciales, ConViT, generación de informes radiológicos: Miura et al. 2021, Hokamp & Liu 2017, Geng et al. 2023). Esa línea de investigación es la que condujo a la decodificación restringida sobre logits y a CCD.
- **El criterio de evaluación:** exigir métricas estándar (precisión, recall, F1, AUC) con intervalos de confianza, en lugar de métricas ad hoc, y descartar enfoques que no se ajustaban al propósito ético del trabajo.
- **La ejecución del experimento principal** en Google Colab (GPU A100), la resolución de los problemas de entorno que aparecieron allí y la validación de los resultados.
- **La interpretación de los resultados**, la estructura y el contenido de la exposición, y las diapositivas (incluidas las secciones de optimización restringida, regularización, monotonía, PINNs y evaluación).

## En qué ayudó la IA

- Escribir y depurar el código del notebook y de los scripts (integración del código oficial de CCD, evaluación con CheXbert, *bootstrap* y gráficas), siguiendo las indicaciones del equipo.
- Correr pruebas exploratorias rápidas (restricciones sobre BiomedCLIP, prueba con un LLM) y verificar contra los artículos originales las cifras y referencias citadas.
- Redactar borradores de la documentación del repositorio y de algunas diapositivas, que el equipo revisó y corrigió.

## Lo que no es nuestro ni de la IA

El método **CCD (Clinical Contrastive Decoding)** es de Zhang, Meng, Lever y Ho (ACL 2026); su código oficial está en [github.com/X-iZhang/CCD](https://github.com/X-iZhang/CCD). Los modelos (MAIRA-2, DenseNet de `torchxrayvision`, CheXbert, BiomedCLIP) y los datos (IU X-Ray, PatchCamelyon, Camelyon17) pertenecen a sus autores y se citan en el [README](README.md#8-referencias).
