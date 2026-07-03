/*
 * 04_expanded_event_screening.js
 *
 * Screens 60 candidate flood events for pseudo-label suitability.
 * Corrected metric vs 03_event_screening.js:
 *
 *   OLD metric: urban_km2 = built-up pixels within S1∩S2 overlap
 *   NEW metric: peri_urban_km2 = pixels within 500m of built-up BUT NOT
 *               themselves built-up, within S1∩S2 overlap
 *
 * Why the correction:
 *   NDWI fires on open water surfaces. In flooded urban areas, water sits
 *   in streets, parks, gardens, car parks — pixels WorldCover classifies
 *   as "herbaceous", "bare", or "road", not "built-up".
 *   The SAR context is still urban (buildings nearby cause shadows and
 *   double-bounce). So peri-urban open land adjacent to buildings is
 *   exactly where pseudo-labels are reliable AND useful for urban SAR
 *   flood detection training.
 *
 * Grade criteria (revised):
 *   A: peri_urban_km2 ≥ 50  AND cloud ≤ 10% AND gap ≤ 2 days
 *   B: peri_urban_km2 ≥ 25  AND cloud ≤ 20% AND gap ≤ 3 days
 *   C: peri_urban_km2 ≥ 10  AND cloud ≤ 30% AND gap ≤ 5 days
 *   D: everything else
 *
 * Events selected to cover:
 *   - Diverse geographies (6 continents)
 *   - Diverse city morphologies (European medieval, Asian megacity,
 *     American suburban sprawl, African informal, Oceania coastal)
 *   - Peri-urban riverine and coastal surge (not dense megacity pluvial)
 *   - Zero overlap with UrbanSARFloods benchmark events
 *
 * UrbanSARFloods events (excluded to preserve clean benchmark):
 *   Houston Harvey 2017, Beira Idai 2019, Western Japan 2018,
 *   Typhoon Hagibis 2019, Sydney 2021/2022, Coraki 2022,
 *   Lokoja Nigeria 2022, Hebei China 2023, Beledweyne Somalia 2023,
 *   Canada 2019, Iran 2019, Lumberton 2016
 *
 * Run in batches of 10 to avoid GEE timeouts:
 *   Set BATCH_START=0,  BATCH_END=10  → run
 *   Set BATCH_START=10, BATCH_END=20  → run
 *   ... etc
 */

// ============================================================
// CONFIGURATION
// ============================================================

var CFG = {
  cloud_threshold:      40,    // generous for screening
  temporal_window_days:  5,
  urban_buffer_m:       500,   // peri-urban zone radius
  scale_m:              200,   // coarser scale for faster screening
  drive_folder:         'water_detection_screening',

  // Grade thresholds (peri_urban_km2)
  grade_A: { peri_km2: 50,  cloud: 10, gap: 2 },
  grade_B: { peri_km2: 25,  cloud: 20, gap: 3 },
  grade_C: { peri_km2: 10,  cloud: 30, gap: 5 },
};

var WORLDCOVER = ee.Image('ESA/WorldCover/v200/2021');
var BUILT_UP   = WORLDCOVER.eq(50);

// Peri-urban zone: within urban_buffer_m of built-up, but NOT built-up itself
// This is computed per-AOI in the screening function to avoid global computation
function getPeriUrban(aoi) {
  var builtUpBuffer = BUILT_UP
    .focal_max({ radius: CFG.urban_buffer_m, units: 'meters',
                 kernelType: 'circle' })
    .clip(aoi);
  return builtUpBuffer.and(BUILT_UP.clip(aoi).not());
}

// ============================================================
// EVENT CATALOGUE — 60 events
// Deliberately excludes all UrbanSARFloods events.
// Focuses on peri-urban riverine and coastal surge.
// ============================================================

var EVENTS = [

  // ── NORTH AMERICA ───────────────────────────────────────────────────────

  {
    id: 1, name: 'Hurricane_Florence_2018',
    lat: 34.2257, lon: -77.9447, buffer_km: 60,
    start: '2018-09-14', end: '2018-09-28',
    country: 'USA', type: 'Coastal+Riverine',
    notes: 'Carolinas suburban sprawl; ARIA FPM+DPM; weeks of inundation'
  },
  {
    id: 2, name: 'Hurricane_Ian_2022',
    lat: 26.6406, lon: -81.8723, buffer_km: 50,
    start: '2022-09-28', end: '2022-10-05',
    country: 'USA', type: 'Coastal',
    notes: 'Fort Myers; S2 usable Sep 30; ARIA DPM; suburban surge'
  },
  {
    id: 3, name: 'Hurricane_Ida_Louisiana_2021',
    lat: 29.6499, lon: -90.7040, buffer_km: 60,
    start: '2021-08-29', end: '2021-09-08',
    country: 'USA', type: 'Coastal+Riverine',
    notes: 'Houma/Lafourche; ARIA DPM; suburban inundation'
  },
  {
    id: 4, name: 'Hurricane_Dorian_Bahamas_2019',
    lat: 26.5420, lon: -77.0639, buffer_km: 30,
    start: '2019-09-01', end: '2019-09-10',
    country: 'Bahamas', type: 'Coastal',
    notes: 'Freeport; EMSR385; ARIA FPM; small island city'
  },
  {
    id: 5, name: 'Hurricane_Maria_Puerto_Rico_2017',
    lat: 18.2208, lon: -66.5901, buffer_km: 80,
    start: '2017-09-20', end: '2017-10-05',
    country: 'Puerto_Rico', type: 'Coastal+Riverine',
    notes: 'Island-wide; ARIA DPM; persistent cloud challenge'
  },
  {
    id: 6, name: 'Midwest_Floods_Nebraska_2019',
    lat: 41.2565, lon: -95.9345, buffer_km: 60,
    start: '2019-03-13', end: '2019-04-15',
    country: 'USA', type: 'Riverine',
    notes: 'Missouri River; Sen1Floods11; spring clear skies'
  },
  {
    id: 7, name: 'Hurricane_Eta_Honduras_2020',
    lat: 15.5000, lon: -88.0000, buffer_km: 60,
    start: '2020-11-03', end: '2020-11-20',
    country: 'Honduras', type: 'Coastal+Riverine',
    notes: 'San Pedro Sula Sula Valley; EMSR477; 6 CEMS activations'
  },
  {
    id: 8, name: 'Rio_Grande_Sul_Brazil_2024',
    lat: -30.0346, lon: -51.2177, buffer_km: 60,
    start: '2024-05-01', end: '2024-05-20',
    country: 'Brazil', type: 'Riverine',
    notes: 'Porto Alegre; EMSR720; 76km2 documented urban flood'
  },
  {
    id: 9, name: 'Recife_Brazil_2022',
    lat: -8.0476, lon: -34.8770, buffer_km: 40,
    start: '2022-05-25', end: '2022-06-05',
    country: 'Brazil', type: 'Riverine',
    notes: 'Recife metropolitan; 100+ deaths; dense coastal city'
  },
  {
    id: 10, name: 'Buenos_Aires_Argentina_2013',
    lat: -34.6037, lon: -58.3816, buffer_km: 40,
    start: '2013-04-02', end: '2013-04-12',
    country: 'Argentina', type: 'Pluvial',
    notes: 'Pre-S1/S2 era — skip if no data found'
  },

  // ── EUROPE ──────────────────────────────────────────────────────────────

  {
    id: 11, name: 'UK_Yorkshire_2015',
    lat: 53.9590, lon: -1.0815, buffer_km: 50,
    start: '2015-12-26', end: '2016-01-05',
    country: 'UK', type: 'Riverine',
    notes: 'York/Leeds; simultaneous S1+S2 Dec 29; Grade A confirmed'
  },
  {
    id: 12, name: 'Western_Europe_Floods_2021',
    lat: 50.5437, lon: 6.8183, buffer_km: 60,
    start: '2021-07-14', end: '2021-07-22',
    country: 'Germany', type: 'Riverine',
    notes: 'Ahr Valley; EMSR517-520; F1=0.9 RAPID validation'
  },
  {
    id: 13, name: 'Emilia_Romagna_Italy_2023',
    lat: 44.2222, lon: 11.8765, buffer_km: 60,
    start: '2023-05-16', end: '2023-05-28',
    country: 'Italy', type: 'Riverine',
    notes: 'Faenza/Forli; EMSR659+664; building-level GT available'
  },
  {
    id: 14, name: 'Venice_Italy_2019',
    lat: 45.4408, lon: 12.3155, buffer_km: 30,
    start: '2019-11-12', end: '2019-11-20',
    country: 'Italy', type: 'Coastal',
    notes: 'Acqua alta 187cm; unique coastal urban morphology'
  },
  {
    id: 15, name: 'Liege_Belgium_2021',
    lat: 50.6326, lon: 5.5797, buffer_km: 30,
    start: '2021-07-14', end: '2021-07-22',
    country: 'Belgium', type: 'Riverine',
    notes: 'Same event as Ahr; EMSR518; Meuse river'
  },
  {
    id: 16, name: 'Paris_Seine_2016',
    lat: 48.8566, lon: 2.3522, buffer_km: 40,
    start: '2016-05-30', end: '2016-06-12',
    country: 'France', type: 'Riverine',
    notes: 'Seine at 6.1m; dense European city; Louvre flooded'
  },
  {
    id: 17, name: 'Thessaloniki_Greece_2023',
    lat: 40.6401, lon: 22.9444, buffer_km: 40,
    start: '2023-09-05', end: '2023-09-12',
    country: 'Greece', type: 'Riverine',
    notes: 'Storm Daniel; Larissa/Thessaloniki; Mediterranean city'
  },
  {
    id: 18, name: 'Prague_Czech_2002',
    lat: 50.0755, lon: 14.4378, buffer_km: 30,
    start: '2002-08-07', end: '2002-08-20',
    country: 'Czech', type: 'Riverine',
    notes: 'Pre-Sentinel era — will return 0 pairs, skip'
  },
  {
    id: 19, name: 'Dublin_Ireland_2015',
    lat: 53.3498, lon: -6.2603, buffer_km: 30,
    start: '2015-12-26', end: '2016-01-05',
    country: 'Ireland', type: 'Riverine',
    notes: 'Storm Desmond; same period as Yorkshire; different city'
  },
  {
    id: 20, name: 'Cologne_Rhine_2021',
    lat: 50.9375, lon: 6.9603, buffer_km: 30,
    start: '2021-07-14', end: '2021-07-22',
    country: 'Germany', type: 'Riverine',
    notes: 'Rhine flooding same event as Ahr; large urban area'
  },

  // ── AFRICA ──────────────────────────────────────────────────────────────

  {
    id: 21, name: 'Cyclone_Idai_Beira_2019',
    lat: -19.8436, lon: 34.8389, buffer_km: 40,
    start: '2019-03-14', end: '2019-03-28',
    country: 'Mozambique', type: 'Coastal+Riverine',
    notes: 'IN UrbanSARFloods — include to check S2 availability'
  },
  {
    id: 22, name: 'KwaZulu_Natal_Durban_2022',
    lat: -29.8587, lon: 30.9810, buffer_km: 50,
    start: '2022-04-09', end: '2022-04-20',
    country: 'South_Africa', type: 'Riverine',
    notes: 'Durban metro; 443 deaths; UNOSAT; African urban morphology'
  },
  {
    id: 23, name: 'Libya_Derna_2023',
    lat: 32.7570, lon: 22.6389, buffer_km: 20,
    start: '2023-09-10', end: '2023-09-20',
    country: 'Libya', type: 'Dam_breach',
    notes: 'Arid clear skies; catastrophic but small city'
  },
  {
    id: 24, name: 'Tunisia_Nabeul_2021',
    lat: 36.4561, lon: 10.7376, buffer_km: 25,
    start: '2021-09-22', end: '2021-09-30',
    country: 'Tunisia', type: 'Flash_flood',
    notes: 'Mediterranean coastal; 5 deaths; clear post-storm'
  },
  {
    id: 25, name: 'Accra_Ghana_2015',
    lat: 5.6037, lon: -0.1870, buffer_km: 30,
    start: '2015-06-03', end: '2015-06-15',
    country: 'Ghana', type: 'Pluvial',
    notes: 'Dense West African coastal city; Sen1Floods11 event'
  },
  {
    id: 26, name: 'Khartoum_Sudan_2020',
    lat: 15.5007, lon: 32.5599, buffer_km: 50,
    start: '2020-08-01', end: '2020-09-01',
    country: 'Sudan', type: 'Riverine',
    notes: 'Nile flooding; arid clear skies; large city'
  },
  {
    id: 27, name: 'Alexandria_Egypt_2015',
    lat: 31.2001, lon: 29.9187, buffer_km: 30,
    start: '2015-10-24', end: '2015-11-01',
    country: 'Egypt', type: 'Flash_flood',
    notes: 'Mediterranean coastal; rare flood; clear skies'
  },

  // ── SOUTH ASIA ──────────────────────────────────────────────────────────

  {
    id: 28, name: 'Chennai_India_2015',
    lat: 13.0827, lon: 80.2707, buffer_km: 50,
    start: '2015-11-25', end: '2015-12-10',
    country: 'India', type: 'Riverine',
    notes: 'Chennai metro; ISRO/NRSC maps; early S1 era'
  },
  {
    id: 29, name: 'Kerala_India_2018',
    lat: 9.9816, lon: 76.2999, buffer_km: 60,
    start: '2018-08-09', end: '2018-08-25',
    country: 'India', type: 'Riverine',
    notes: 'Kochi; 10+ SAR papers; monsoon cloud challenge'
  },
  {
    id: 30, name: 'Cyclone_Amphan_Kolkata_2020',
    lat: 22.5726, lon: 88.3639, buffer_km: 70,
    start: '2020-05-20', end: '2020-06-05',
    country: 'India', type: 'Coastal',
    notes: 'Kolkata; storm surge drains fast; S1 May 22 best bet'
  },
  {
    id: 31, name: 'Pakistan_Sindh_2022',
    lat: 27.5260, lon: 68.7732, buffer_km: 80,
    start: '2022-08-10', end: '2022-09-15',
    country: 'Pakistan', type: 'Riverine',
    notes: 'Massive extent; TU Wien open labels; Indus overflow'
  },
  {
    id: 32, name: 'Assam_India_2020',
    lat: 26.2006, lon: 92.9376, buffer_km: 50,
    start: '2020-07-01', end: '2020-07-31',
    country: 'India', type: 'Riverine',
    notes: 'Guwahati; Brahmaputra; annual but severe 2020'
  },
  {
    id: 33, name: 'Bangladesh_Sylhet_2022',
    lat: 24.8949, lon: 91.8687, buffer_km: 40,
    start: '2022-06-15', end: '2022-07-01',
    country: 'Bangladesh', type: 'Riverine',
    notes: 'Surma River; worst in 20 years; dense low-lying city'
  },
  {
    id: 34, name: 'Karachi_Pakistan_2020',
    lat: 24.8607, lon: 67.0011, buffer_km: 50,
    start: '2020-08-27', end: '2020-09-05',
    country: 'Pakistan', type: 'Pluvial',
    notes: 'Record monsoon; dense coastal megacity; arid baseline'
  },
  {
    id: 35, name: 'Hyderabad_India_2020',
    lat: 17.3850, lon: 78.4867, buffer_km: 40,
    start: '2020-10-13', end: '2020-10-22',
    country: 'India', type: 'Pluvial',
    notes: 'Musi River; 50+ deaths; tech city suburban flooding'
  },

  // ── SOUTHEAST ASIA ──────────────────────────────────────────────────────

  {
    id: 36, name: 'Jakarta_Indonesia_2020',
    lat: -6.2088, lon: 106.8456, buffer_km: 40,
    start: '2020-01-01', end: '2020-01-10',
    country: 'Indonesia', type: 'Riverine',
    notes: '17% of Jakarta flooded; ARIA FPM; dense megacity'
  },
  {
    id: 37, name: 'Typhoon_Vamco_Manila_2020',
    lat: 14.6760, lon: 121.0437, buffer_km: 40,
    start: '2020-11-11', end: '2020-11-18',
    country: 'Philippines', type: 'Coastal+Riverine',
    notes: 'Marikina River; 804 urban km2 in screening; Grade C'
  },
  {
    id: 38, name: 'Thailand_Bangkok_2011',
    lat: 13.7563, lon: 100.5018, buffer_km: 60,
    start: '2011-10-01', end: '2011-11-30',
    country: 'Thailand', type: 'Riverine',
    notes: 'Pre-Sentinel era — will return 0 pairs'
  },
  {
    id: 39, name: 'Vietnam_Hue_2020',
    lat: 16.4637, lon: 107.5909, buffer_km: 30,
    start: '2020-10-06', end: '2020-10-20',
    country: 'Vietnam', type: 'Riverine',
    notes: 'Huong River; Central Vietnam floods; historic city'
  },
  {
    id: 40, name: 'Malaysia_KL_2021',
    lat: 3.1390, lon: 101.6869, buffer_km: 40,
    start: '2021-12-17', end: '2021-12-28',
    country: 'Malaysia', type: 'Pluvial',
    notes: 'Klang River; 50-year flood; KL suburbs inundated'
  },
  {
    id: 41, name: 'Myanmar_Bago_2021',
    lat: 17.3350, lon: 96.4800, buffer_km: 40,
    start: '2021-08-07', end: '2021-08-20',
    country: 'Myanmar', type: 'Riverine',
    notes: 'Cyclone Yagi precursor; Bago River; peri-urban'
  },

  // ── EAST ASIA ───────────────────────────────────────────────────────────

  {
    id: 42, name: 'Zhengzhou_China_2021',
    lat: 34.7472, lon: 113.6249, buffer_km: 40,
    start: '2021-07-17', end: '2021-07-31',
    country: 'China', type: 'Riverine+Pluvial',
    notes: 'Grade A in screening; 365 urban km2; megacity'
  },
  {
    id: 43, name: 'Hebei_China_2023',
    lat: 39.3054, lon: 115.4000, buffer_km: 60,
    start: '2023-07-28', end: '2023-08-15',
    country: 'China', type: 'Riverine',
    notes: 'IN UrbanSARFloods — listed to compare S2 availability'
  },
  {
    id: 44, name: 'Seoul_South_Korea_2022',
    lat: 37.5665, lon: 126.9780, buffer_km: 40,
    start: '2022-08-08', end: '2022-08-15',
    country: 'South_Korea', type: 'Pluvial',
    notes: 'Record 24hr rainfall; Gangnam; dense city'
  },
  {
    id: 45, name: 'Osaka_Japan_2018',
    lat: 34.6937, lon: 135.5023, buffer_km: 40,
    start: '2018-07-05', end: '2018-07-15',
    country: 'Japan', type: 'Riverine',
    notes: 'July 2018 Japan floods; multiple cities; Kinki region'
  },
  {
    id: 46, name: 'Typhoon_Hagibis_Tokyo_2019',
    lat: 35.6762, lon: 139.6503, buffer_km: 60,
    start: '2019-10-12', end: '2019-10-20',
    country: 'Japan', type: 'Coastal+Riverine',
    notes: 'ARIA Figshare open dataset; S1 within 12hr of landfall'
  },
  {
    id: 47, name: 'Nagoya_Japan_2000',
    lat: 35.1815, lon: 136.9066, buffer_km: 40,
    start: '2000-09-11', end: '2000-09-20',
    country: 'Japan', type: 'Riverine',
    notes: 'Pre-Sentinel — will return 0 pairs'
  },

  // ── OCEANIA ─────────────────────────────────────────────────────────────

  {
    id: 48, name: 'Lismore_Australia_2022',
    lat: -28.8133, lon: 153.2750, buffer_km: 25,
    start: '2022-02-28', end: '2022-03-12',
    country: 'Australia', type: 'Riverine',
    notes: 'EMSR570+637; levee overtopped 2.12m above record; CBD submerged'
  },
  {
    id: 49, name: 'Cyclone_Gabrielle_NZ_2023',
    lat: -39.4928, lon: 176.9120, buffer_km: 50,
    start: '2023-02-12', end: '2023-02-22',
    country: 'New_Zealand', type: 'Coastal+Riverine',
    notes: 'Napier/Hastings; Dragonfly SAR polygons; Grade C confirmed'
  },
  {
    id: 50, name: 'Perth_Australia_2021',
    lat: -31.9505, lon: 115.8605, buffer_km: 40,
    start: '2021-05-28', end: '2021-06-10',
    country: 'Australia', type: 'Riverine',
    notes: 'Swan River; rare Perth flooding; clear Mediterranean climate'
  },

  // ── MIDDLE EAST ─────────────────────────────────────────────────────────

  {
    id: 51, name: 'Dubai_UAE_2024',
    lat: 25.2048, lon: 55.2708, buffer_km: 35,
    start: '2024-04-14', end: '2024-04-22',
    country: 'UAE', type: 'Flash_flood',
    notes: 'Exceptional clear skies; modern urban; rare flood event'
  },
  {
    id: 52, name: 'Jeddah_Saudi_2022',
    lat: 21.4858, lon: 39.1925, buffer_km: 30,
    start: '2022-11-24', end: '2022-12-03',
    country: 'Saudi_Arabia', type: 'Flash_flood',
    notes: 'Arid climate; clear skies; dense coastal city'
  },
  {
    id: 53, name: 'Oman_Muscat_2024',
    lat: 23.5880, lon: 58.3829, buffer_km: 30,
    start: '2024-04-14', end: '2024-04-22',
    country: 'Oman', type: 'Flash_flood',
    notes: 'Same cyclone as Dubai; arid; coastal city'
  },

  // ── CENTRAL ASIA / CAUCASUS ─────────────────────────────────────────────

  {
    id: 54, name: 'Tbilisi_Georgia_2015',
    lat: 41.6938, lon: 44.8015, buffer_km: 25,
    start: '2015-06-13', end: '2015-06-22',
    country: 'Georgia', type: 'Flash_flood',
    notes: 'Vere River; zoo animals escaped; compact European city'
  },
  {
    id: 55, name: 'Kazakhstan_Aktobe_2024',
    lat: 50.2839, lon: 57.1670, buffer_km: 40,
    start: '2024-04-05', end: '2024-04-25',
    country: 'Kazakhstan', type: 'Riverine',
    notes: 'Ural/Ilek rivers; worst in 80 years; steppe city'
  },

  // ── ADDITIONAL EUROPE ────────────────────────────────────────────────────

  {
    id: 56, name: 'Seville_Spain_2025',
    lat: 37.3891, lon: -5.9845, buffer_km: 40,
    start: '2025-01-28', end: '2025-02-05',
    country: 'Spain', type: 'Riverine',
    notes: 'Guadalquivir; Jan 2025; dense Andalusian city'
  },
  {
    id: 57, name: 'Valencia_Spain_2024',
    lat: 39.4699, lon: -0.3763, buffer_km: 30,
    start: '2024-10-29', end: '2024-11-10',
    country: 'Spain', type: 'Flash_flood',
    notes: 'DANA floods; peri-urban towns south of Valencia'
  },
  {
    id: 58, name: 'Famagusta_Cyprus_2019',
    lat: 35.1284, lon: 33.9395, buffer_km: 20,
    start: '2019-01-10', end: '2019-01-20',
    country: 'Cyprus', type: 'Flash_flood',
    notes: 'Small Mediterranean city; rare event; clear skies'
  },
  {
    id: 59, name: 'Bucharest_Romania_2024',
    lat: 44.4268, lon: 26.1025, buffer_km: 40,
    start: '2024-09-13', end: '2024-09-22',
    country: 'Romania', type: 'Flash_flood',
    notes: 'Storm Boris; Galati region; Eastern European city'
  },
  {
    id: 60, name: 'Vojvodina_Serbia_2014',
    lat: 45.2671, lon: 19.8335, buffer_km: 40,
    start: '2014-05-14', end: '2014-05-28',
    country: 'Serbia', type: 'Riverine',
    notes: 'Sava/Danube; Obrenovac; pre-S2 but S1A launched Oct 2014'
  },
];

// ============================================================
// CORE SCREENING FUNCTION
// ============================================================

function screenEvent(event) {
  var aoi = ee.Geometry.Point([event.lon, event.lat])
              .buffer(event.buffer_km * 1000);
  var windowMs = CFG.temporal_window_days * 24 * 3600 * 1000;

  var s1Col = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(aoi)
    .filterDate(event.start, event.end)
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .filter(ee.Filter.eq('instrumentMode', 'IW'));

  var s2Col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(aoi)
    .filterDate(event.start, event.end)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CFG.cloud_threshold));

  var joined = ee.Join.saveAll({ matchesKey: 's2_matches' })
    .apply(
      s1Col, s2Col,
      ee.Filter.and(
        ee.Filter.intersects({ leftField: '.geo', rightField: '.geo' }),
        ee.Filter.maxDifference({
          difference:  windowMs,
          leftField:   'system:time_start',
          rightField:  'system:time_start'
        })
      )
    );

  var withBest = joined.map(function(s1img) {
    var matches = ee.ImageCollection.fromImages(s1img.get('s2_matches'));
    var bestS2  = matches.sort('CLOUDY_PIXEL_PERCENTAGE').first();
    var gapDays = ee.Number(s1img.get('system:time_start'))
                    .subtract(ee.Number(bestS2.get('system:time_start')))
                    .abs().divide(86400000);
    return s1img
      .set('best_s2',      bestS2)
      .set('best_cloud',   bestS2.get('CLOUDY_PIXEL_PERCENTAGE'))
      .set('gap_days',     gapDays)
      .set('n_s2_matches', matches.size());
  });

  var valid = withBest
    .filter(ee.Filter.gt('n_s2_matches', 0))
    .sort('best_cloud');

  return { valid: valid, s1Count: s1Col.size(), s2Count: s2Col.size(), aoi: aoi };
}

function computeStats(best, aoi) {
  var bestS2  = ee.Image(best.get('best_s2'));
  var overlap = best.geometry()
                  .intersection(bestS2.geometry(), ee.ErrorMargin(10))
                  .intersection(aoi, ee.ErrorMargin(10));

  // Guard: if intersection is empty, return zeros instead of crashing
  var overlapAreaM2 = overlap.area(10);
  var isEmpty = overlapAreaM2.lt(1000);  // < 1000 m² = effectively empty

  // Use a safe fallback geometry (single point) when overlap is empty
  // so .clip() and .reduceRegion() don't throw
  var safeOverlap = ee.Algorithms.If(
    isEmpty,
    ee.Geometry.Point([0, 0]).buffer(10),  // tiny placeholder
    overlap
  );
  safeOverlap = ee.Geometry(safeOverlap);

  var periUrban  = getPeriUrban(safeOverlap);
  var overlapKm2 = overlapAreaM2.divide(1e6);

  var periKm2Raw = periUrban.reduceRegion({
    reducer:   ee.Reducer.sum(),
    geometry:  safeOverlap,
    scale:     CFG.scale_m,
    maxPixels: 1e11
  }).get('Map');

  var pixelKm2  = ee.Number(CFG.scale_m).pow(2).divide(1e6);
  // If overlap was empty, force peri_km2 to 0
  var periKm2Val = ee.Algorithms.If(
    isEmpty,
    ee.Number(0),
    ee.Number(periKm2Raw).multiply(pixelKm2)
  );

  return {
    overlapKm2: overlapKm2,
    periKm2:    ee.Number(periKm2Val),
    bestS2:     bestS2,
    overlap:    overlap
  };
}

function assignGrade(cloud, gap, periKm2) {
  var A = CFG.grade_A, B = CFG.grade_B, C = CFG.grade_C;
  if (periKm2 >= A.peri_km2 && cloud <= A.cloud && gap <= A.gap) return 'A';
  if (periKm2 >= B.peri_km2 && cloud <= B.cloud && gap <= B.gap) return 'B';
  if (periKm2 >= C.peri_km2 && cloud <= C.cloud && gap <= C.gap) return 'C';
  return 'D';
}

function pad(val, w) {
  var s = String(val);
  while (s.length < w) s = s + ' ';
  return s.substring(0, w);
}

// ============================================================
// RUN — process in batches of 10
// ============================================================

var BATCH_START = 0;
var BATCH_END   = 10;

var batch = EVENTS.slice(BATCH_START, BATCH_END);

print('================================================');
print('Expanded event screening (' + batch.length + ' events)');
print('Batch: ' + BATCH_START + ' – ' + BATCH_END);
print('Metric: peri-urban km² (500m buffer, non-built-up)');
print('================================================');
print('');
print('ID  | Event                          | S1 | S2 | Pairs | S1 Date    | S2 Date    | Cloud% | Gap(d) | Overlap km² | PeriUrban km² | Grade | Notes');
print('----|--------------------------------|----|----|----- -|------------|------------|--------|--------|-------------|---------------|-------|------');

var results = [];

batch.forEach(function(event) {
  var r = screenEvent(event);
  var s1Count = r.s1Count.getInfo();
  var s2Count = r.s2Count.getInfo();
  var nPairs  = r.valid.size().getInfo();

  if (nPairs === 0) {
    print(
      pad(event.id, 4) + ' | ' + pad(event.name, 30) + ' | ' +
      pad(s1Count, 2) + ' | ' + pad(s2Count, 2) + ' | ' +
      pad(0, 5) + ' | N/A        | N/A        | N/A    | N/A    | ' +
      'N/A         | N/A           | D     | No pairs — ' + event.notes
    );
    results.push(ee.Feature(null, {
      id: event.id, name: event.name, country: event.country,
      type: event.type, s1_count: s1Count, s2_count: s2Count,
      n_pairs: 0, s1_date: 'N/A', s2_date: 'N/A',
      cloud_pct: -1, gap_days: -1, overlap_km2: 0,
      peri_urban_km2: 0, grade: 'D', notes: event.notes
    }));
    return;
  }

  var best   = ee.Image(r.valid.first());
  var stats  = computeStats(best, r.aoi);

  var cloud      = ee.Number(best.get('best_cloud')).round().getInfo();
  var gap        = ee.Number(best.get('gap_days')).round().getInfo();
  var overlapKm2 = stats.overlapKm2.round().getInfo();
  var periKm2    = stats.periKm2.round().getInfo();
  var s1Date     = best.date().format('YYYY-MM-dd').getInfo();
  var s2Date     = stats.bestS2.date().format('YYYY-MM-dd').getInfo();
  var grade      = assignGrade(cloud, gap, periKm2);

  print(
    pad(event.id, 4) + ' | ' + pad(event.name, 30) + ' | ' +
    pad(s1Count, 2) + ' | ' + pad(s2Count, 2) + ' | ' +
    pad(nPairs, 5) + ' | ' + pad(s1Date, 10) + ' | ' + pad(s2Date, 10) + ' | ' +
    pad(cloud, 6) + ' | ' + pad(gap, 6) + ' | ' +
    pad(overlapKm2, 11) + ' | ' + pad(periKm2, 13) + ' | ' +
    pad(grade, 5) + ' | ' + event.notes
  );

  results.push(ee.Feature(null, {
    id: event.id, name: event.name, country: event.country,
    type: event.type, s1_count: s1Count, s2_count: s2Count,
    n_pairs: nPairs, s1_date: s1Date, s2_date: s2Date,
    cloud_pct: cloud, gap_days: gap,
    overlap_km2: overlapKm2, peri_urban_km2: periKm2,
    grade: grade, notes: event.notes
  }));
});

// Export CSV for this batch
Export.table.toDrive({
  collection:    ee.FeatureCollection(results),
  description:   'expanded_screening_batch_' + BATCH_START + '_' + BATCH_END,
  folder:        CFG.drive_folder,
  fileNamePrefix: 'expanded_screening_' + BATCH_START + '_' + BATCH_END,
  fileFormat:    'CSV'
});

print('');
print('CSV queued → Tasks tab → Run');
print('Change BATCH_START/BATCH_END to screen next 10 events.');
