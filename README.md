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

### [NYC Taxi Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

Characteristics:

- Period covered: January 1, 2009 – June 30, 2010
- Time steps: 7,098
- Time range: 8:00 a.m. – 8:00 p.m.
- Number of records: 92 million
- Target variable: Pickup demand

### [Data de UBER NYC](https://ieee-dataport.org/documents/uber-pickups-new-york-city)

Characteristics:

- Period covered: April 1, 2014 – September 30, 2014
- Number of records: 1.5 million
- Time steps: 2,379
- Time range: 8:00 a.m. – 8:00 p.m.
- Target variable: Pickup demand

### Data de CTL

Characteristics:

- Period covered: April 1, 2020 – July 25, 2020
- Number of records: 0.2 million
- Time steps: 1,880
- Time range: 8:00 a.m. – 8:00 p.m.
- Target variable: Traffic speed

## Proposed Method
