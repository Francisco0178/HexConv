# HexConv - A ConvLSTM NN for geospatio-temporal hexagonal based data prediction

![HexConv](imgs/hexconv2.png)

Este trabajo define una metodología de preprocesamiento hexagonal

- [Introduction](#introduction)
- [Related Work](#related-work)
- [Data Used](#data-used)
- [Proposed Method](#proposed-method)
  - [Why Hexagons](#why-hexagons)
  - [S2 vs H3](#s2-vs-h3)
- [Data Preprocessing](#data-preprocessing)
- [Imputation](#imputation)
- [HexConvLSTM Definition](#hexconvlstm-definition)
- [Data Postprocessing](#data-postprocessing)
- [Results](#results)
- [Conclusion](#conclusion)
- [Acknowledgements](#acknowledgements)
- [Citing HexConv](#citing-hexconv)

## Introduction

As an introduction, this work was motivated by the [Centro de Transporte y Logística de la Universidad Andrés Bello](https://ctl.unab.cl/) (CTL) in collaboration with the [Departamento de Ciencias de la Computación de la Pontifica Universidad Católica](https://dcc.ing.uc.cl/) and [CENIA](https://cenia.cl/investigacion/).

The goal of this study is to design and develop a neural network capable of receiving hexagonally aggregated input data for the estimation of spatio-temporal variables such as traffic speed and passenger transport demand.

The neural network selected for this research is a Convolutional LSTM network. Although this model does not naturally handle hexagonal data, it offers significant advantages for estimating variables that change over both time and space, such as in videos, satellite imagery, or GPS-collected data.

In this work, we preprocess and adapt GPS data so that a ConvLSTM network can logically interpret the structure of a hexagonal topology embedded within a square grid, which is subsequently processed through a specialized kernel designed to capture the original neighbors of each hexagon.

## Related work

Un trabajo relacionado en la incorporación de hexágonos en redes neuronales profundas es el de [Hexagldy](https://github.com/ai4iacts/hexagdly), en este trabajo... mencionar cómo y para qué se aprovecharon los hexágonos aquí.

## Data Used

For this research, three datasets were used — two publicly available and one private:

| Dataset           | Period Covered             | Records     | Time Steps | Time Range            | Target Variable | Link                                                                   |
| ----------------- | -------------------------- | ----------- | ---------- | --------------------- | --------------- | ---------------------------------------------------------------------- |
| **NYC Taxi Data** | Jan 1, 2009 – Jun 30, 2010 | 92 million  | 7,098      | 8:00 a.m. – 8:00 p.m. | Pickup demand   | [Link](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)   |
| **UBER NYC Data** | Apr 1, 2014 – Sep 30, 2014 | 1.5 million | 2,379      | 8:00 a.m. – 8:00 p.m. | Pickup demand   | [Link](https://ieee-dataport.org/documents/uber-pickups-new-york-city) |
| **CTL Data**      | Apr 1, 2020 – Jul 25, 2020 | 0.2 million | 1,880      | 8:00 a.m. – 8:00 p.m. | Traffic speed   | Private                                                                |

Para cada una de las zonas utilizadas para este trabajo (Nueva York y Santiago) se determinó una subregión de datos densa. Esto se hizo con el fin de minimzar lo máximo posible áreas que pudieran contener muchos datos faltantes. Cabe recordar que la agrupación de los datos además de ser espacial también es temporal, lo que se traduce en que debemos asegurar registros para cada hora y para cada hexágono. Las subregiones escogidas fueron las siguientes:

<p align="center">
  <img src="/imgs/nyczone.png" alt="Imagen 1">
  <img src="/imgs/ctlzone.png" alt="Imagen 2">
</p>

## Proposed Method

The proposed method for this work involves indexing the data into hexagonal cells, preprocessing and properly adapting the data to the hexagonal pipeline through a series of matrix operations, data imputation, and the definition of a neural network combined with a specialized kernel for hexagonal convolution. The data are then trained as image sequences at a resolution determined by the number of hexagons.

### Why Hexagons?

Before proceeding, it is important to justify the decision to explore hexagonal convolutions instead of conventional grid-based convolutions.

A regular tessellation refers to a way of completely covering a plane using only one type of regular polygon, without leaving any gaps or overlaps between shapes. There are three possible regular tessellations: triangles, squares, and hexagons. The main reasons for choosing hexagons are neighbor traversal, subdivision, and distortion.

#### Neighbour traversal

Neighbor traversal refers to the process of exploring or iterating through the neighboring cells surrounding a given cell within a tessellation or grid. This process is fundamental in simulations, spatial models, and algorithms that rely on interactions between adjacent cells.

In this context, neighbor traversal is more convenient in hexagonal tessellations because each cell has six neighbors located at an equal distance. This uniformity simplifies mathematical computations such as distance calculations, averaging, and value propagation, making them more precise and consistent. Unlike triangular or square tessellations—which involve multiple classes of neighbors at varying distances—the hexagonal pattern maintains a balanced and homogeneous geometry, eliminating the need to distinguish between different neighbor types. As a result, algorithmic complexity is reduced and efficiency in simulations and computational models is improved. Consequently, neighbor traversal becomes more direct, coherent, and easier to implement in hexagonal tessellations.

#### Subdivision

Subdivision using hexagons is more advantageous because, although they cannot be perfectly divided like squares, their geometry allows for highly accurate approximations of areas of different sizes through rotation and alternation. This makes them ideal for representing variable scales in spatial models.

This characteristic complements the advantage of neighbor traversal, as both aspects promote mathematical uniformity and consistency: hexagons maintain equidistant spacing between neighboring cells and allow regions to be merged or subdivided without distorting the structure. Furthermore, their shape facilitates processes such as sharding, enabling efficient and continuous data partitioning across multiple levels of resolution. Overall, the hexagonal pattern provides a balanced trade-off between precision, flexibility, and computational simplicity.

#### Distorsion

Distortion is one of the most significant reasons for choosing hexagons in spatial indexing, since projecting the spherical surface of the Earth onto a plane inevitably introduces deformation. Hexagons—particularly when combined with structures such as the icosahedron and the Dymaxion projection—effectively minimize this distortion, preserving a more uniform and stable shape compared to squares or cubes. This leads to a more accurate representation of geographic areas, enhances visual coherence, and ensures a more balanced distribution of data across the plane, which is essential for modeling, geospatial analysis, and visualization applications.

### H3 vs S2

Para la indexación hexagonal se utilizo la librería [H3 de Uber](https://h3geo.org/).
