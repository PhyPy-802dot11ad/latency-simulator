# PHY IEEE 802.11ad latency simulator

The package is intended for simulating PHY package latency in the IEEE 802.11ad physical layer (PHY).
All data formats closely comply with the standard, while the component performance metrics are derived from literature on integrated circuit (IC) design.

Latency is defined from the top of the transmitter's PHY to the the MAC-PHY interface in the receiver, as illustrated below:

![PHY latency definition][21]

The supported modulation and coding schemes (MCSs) are: 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12.1, 12.3, 12.4, 12.5

Apart from tracking the latency of individual PHY payloads, it also allows the identification of bottlenecks in the RX digital baseband (DBB).
This is achieved by post-processing the descriptive logs generated during simulation.


## Simulation examples
These contain use cases where the simulation framework was used to evaluate PHY latency. The two are intended for:
- Evaluating latency for sequence lengths 1-262 KB and conducting 1-10 decoding iterations
- Studying latency during the transmission of sequences with a specific length, 100 B by default, while conducting 1-100 decoding iterations

## Related work
The ideal case latency calculation module, IEEE 802.11ad component definitions, and the BER simulator repositories reside on [the PhyPy-802dot11ad page][1]. 

## Further reading
- [A. Marinšek and L. Van der Perre (2021) Keeping up with the bits: tracking physical layer latency in millimeter-wave Wi-Fi networks][11]

[1]: https://github.com/PhyPy-802dot11ad
[11]: https://arxiv.org/abs/2105.13147
[21]: ./MPDU_latency_definition.png