# Sensor Characteristics and Considerations

## Incidence Angle Effects

### Why it matters
- Backscatter intensity varies with incidence angle
- Steeper angles (20-25deg): Stronger backscatter overall
- Shallower angles (40-45deg): Weaker backscatter, more shadow in urban areas
- Water appears dark at all angles (specular reflection), but contrast with surroundings changes

### Sentinel-1 specifics
- IW mode: 29.1 to 46.0 deg across swath
- Ascending vs Descending: Buildings shadow opposite directions
- Relative orbit number: Same orbit = consistent geometry

### ALOS-2 ScanSAR
- Variable incidence across wide swath (350km)
- Less precise geometric consistency than Sentinel-1

### Mitigation strategies
1. Normalize: Include incidence angle as input feature to model
2. Standardize: Filter to consistent orbit/angle range (simpler but loses data)
3. Augment: Train on multiple geometries explicitly
4. Ensemble: Separate models per geometry, ensemble at inference

## C-band vs L-band

| Property | C-band (S1) | L-band (ALOS-2) |
|----------|-------------|-----------------|
| Wavelength | 5.6 cm | 23.6 cm |
| Vegetation penetration | Low | High |
| Urban sensitivity | High (surface detail) | Moderate |
| Flood under trees | Poor | Better |
| Double-bounce intensity | Higher | Lower |

### Implications for urban flooding
- C-band: Strong double-bounce from building-water interfaces (bright return)
- L-band: More penetration, potentially different urban signature
- Both struggle with: smooth pavement vs water distinction

## Resolution Alignment

For training with optical pseudo-labels:

| Pairing | Resolution match | Recommended approach |
|---------|------------------|---------------------|
| S2 to S1 | 10m to 10m | Direct pixel alignment |
| S2 to ALOS2 | 10m to 25m | Upscale S2 labels or downsample both |
| L8/9 to S1 | 30m to 10m | Use S2 preferentially; or downsample S1 |
| L8/9 to ALOS2 | 30m to 25m | Reasonable match |

## Temporal Considerations

| Flood type | Persistence | Temporal tolerance |
|------------|-------------|-------------------|
| Flash flood | Hours | Need less than 12hr coincidence |
| Fluvial (river) | Days-weeks | 24-48hr acceptable |
| Coastal/tidal | Tidal cycle | Need tidal phase matching |
| Post-hurricane standing water | Days | 24-48hr acceptable |

## Urban Masking Options in GEE

1. ESA WorldCover 2021: ESA/WorldCover/v200/2021, Class 50 = Built-up, 10m
2. GHSL Built-up: JRC/GHSL/P2023A/GHS_BUILT_S, Multi-temporal, 10m  
3. Dynamic World: GOOGLE/DYNAMICWORLD/V1, Near-real-time, 10m

Recommendation: ESA WorldCover for consistency; Dynamic World if you need flood-time land cover.
