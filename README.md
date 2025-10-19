# HexConv - A ConvLSTM NN for geospatio-temporal hexagonal based data prediction

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

| Dataset           | Period Covered             | Records     | Time Steps | Time Range            | Target Variable | Link                                                                      |
| ----------------- | -------------------------- | ----------- | ---------- | --------------------- | --------------- | ------------------------------------------------------------------------- |
| **NYC Taxi Data** | Jan 1, 2009 – Jun 30, 2010 | 92 million  | 7,098      | 8:00 a.m. – 8:00 p.m. | Pickup demand   | [🌐 Link](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)   |
| **UBER NYC Data** | Apr 1, 2014 – Sep 30, 2014 | 1.5 million | 2,379      | 8:00 a.m. – 8:00 p.m. | Pickup demand   | [🌐 Link](https://ieee-dataport.org/documents/uber-pickups-new-york-city) |
| **CTL Data**      | Apr 1, 2020 – Jul 25, 2020 | 0.2 million | 1,880      | 8:00 a.m. – 8:00 p.m. | Traffic speed   | 🔒 Private                                                                |

## Proposed Method

El método propuesto para este trabajo es indexar los datos en hexágonos, preprocesar y adaptar los datos adecuadamente al pipeline hexagonal a través de una serie de operaciones matriciales, imputación de datos, definición de una red neuronal junto con un kernel especializado para la convolución hexagonal y entrenar los datos como series de imágenes de una resolución determinada por la cantidad de hexágonos.

### Why Hexagons?

Antes de continuar, es importante justificar por qué se decidió explorar convoluciones hexagonales en lugar de convoluciones normales basadas en cuadrículas.

Una teselación regular es una forma de cubrir completamente un plano usando un solo tipo de polígono regular, sin dejar espacios vacíos ni que las figuras se superpongan. Existen 3 teselaciones de polígonos regulares: triángulos, cuadrados y hexágonos. Los motivos principales por los cuales se escogieron los hexagonos son: Neighbour Traversal, Subdivisión y Distorsión.

#### Neighbour traversal
