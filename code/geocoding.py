import re
from typing import Optional

SPATIAL_PHRASES = {
    "near", "nearby", "close to", "not far from", "not too far from", "around", "in the area of", "in the vicinity of",
    "within reach of", "surrounding", "in the greater area", "in the region of", "based around", "located near",
    "headquartered near", "outside of", "just outside", "on the outskirts of", "in the neighborhood of",
    "within driving distance of", "a short distance from", "within 50km of", "within 100km of", "within 50 miles of",
    "within 100 miles of", "less than 50km from",
}

DEFAULT_RADIUS_KM = 50

CITIES: dict[str, tuple[float, float]] = {
    # North America
    "new york": (40.7128, -74.0060), "new york city": (40.7128, -74.0060), "los angeles": (34.0522, -118.2437), "chicago": (41.8781, -87.6298),
    "houston": (29.7604, -95.3698), "phoenix": (33.4484, -112.0740), "philadelphia": (39.9526, -75.1652), "san antonio": (29.4241, -98.4936),
    "san diego": (32.7157, -117.1611), "dallas": (32.7767, -96.7970), "san jose": (37.3382, -121.8863), "austin": (30.2672, -97.7431),
    "san francisco": (37.7749, -122.4194), "seattle": (47.6062, -122.3321), "denver": (39.7392, -104.9903), "boston": (42.3601, -71.0589),
    "nashville": (36.1627, -86.7816), "portland": (45.5051, -122.6750), "las vegas": (36.1699, -115.1398), "atlanta": (33.7490, -84.3880),
    "miami": (25.7617, -80.1918), "minneapolis": (44.9778, -93.2650), "detroit": (42.3314, -83.0458), "charlotte": (35.2271, -80.8431),
    "washington": (38.9072, -77.0369), "washington dc": (38.9072, -77.0369), "pittsburgh": (40.4406, -79.9959), "baltimore": (39.2904, -76.6122),
    "salt lake city": (40.7608, -111.8910), "raleigh": (35.7796, -78.6382), "silicon valley": (37.3875, -122.0575), "palo alto": (37.4419, -122.1430),
    "toronto": (43.6532, -79.3832), "montreal": (45.5017, -73.5673), "vancouver": (49.2827, -123.1207), "calgary": (51.0447, -114.0719),
    "ottawa": (45.4215, -75.6972), "mexico city": (19.4326, -99.1332),

    # Europe — Western
    "london": (51.5074, -0.1278), "manchester": (53.4808, -2.2426), "birmingham": (52.4862, -1.8904), "edinburgh": (55.9533, -3.1883),
    "glasgow": (55.8642, -4.2518), "bristol": (51.4545, -2.5879), "cambridge": (52.2053, 0.1218), "oxford": (51.7520, -1.2577),
    "paris": (48.8566, 2.3522), "lyon": (45.7640, 4.8357), "marseille": (43.2965, 5.3698), "toulouse": (43.6047, 1.4442),
    "nice": (43.7102, 7.2620), "nantes": (47.2184, -1.5536), "strasbourg": (48.5734, 7.7521), "bordeaux": (44.8378, -0.5792),
    "berlin": (52.5200, 13.4050), "munich": (48.1351, 11.5820), "münchen": (48.1351, 11.5820), "hamburg": (53.5511, 9.9937),
    "frankfurt": (50.1109, 8.6821), "cologne": (50.9375, 6.9603), "köln": (50.9375, 6.9603), "düsseldorf": (51.2277, 6.7735),
    "stuttgart": (48.7758, 9.1829), "dortmund": (51.5136, 7.4653), "leipzig": (51.3397, 12.3731), "dresden": (51.0504, 13.7373),
    "amsterdam": (52.3676, 4.9041), "rotterdam": (51.9244, 4.4777), "the hague": (52.0705, 4.3007), "eindhoven": (51.4416, 5.4697), 
    "brussels": (50.8503, 4.3517), "antwerp": (51.2194, 4.4025), "zurich": (47.3769, 8.5417), "zürich": (47.3769, 8.5417),
    "geneva": (46.2044, 6.1432), "basel": (47.5596, 7.5886), "bern": (46.9480, 7.4474), "vienna": (48.2082, 16.3738), 
    "wien": (48.2082, 16.3738), "madrid": (40.4168, -3.7038), "barcelona": (41.3874, 2.1686), "valencia": (39.4699, -0.3763),
    "seville": (37.3891, -5.9845), "lisbon": (38.7223, -9.1393), "porto": (41.1579, -8.6291), "rome": (41.9028, 12.4964),
    "milan": (45.4642, 9.1900), "naples": (40.8518, 14.2681), "turin": (45.0703, 7.6869), "florence": (43.7696, 11.2558), 
    "bologna": (44.4949, 11.3426), "dublin": (53.3498, -6.2603), "cork": (51.8985, -8.4756),

    # Europe — Nordic
    "stockholm": (59.3293, 18.0686), "gothenburg": (57.7089, 11.9746), "malmö": (55.6050, 13.0038), "oslo": (59.9139, 10.7522), 
    "bergen": (60.3913, 5.3221), "copenhagen": (55.6761, 12.5683), "aarhus": (56.1629, 10.2039), "helsinki": (60.1699, 24.9384),
    "tampere": (61.4978, 23.7610), "reykjavik": (64.1466, -21.9426),

    # Europe — Eastern / Central
    "warsaw": (52.2297, 21.0122), "krakow": (50.0647, 19.9450), "wroclaw": (51.1079, 17.0385), "gdansk": (54.3520, 18.6466),
    "prague": (50.0755, 14.4378), "brno": (49.1951, 16.6068), "budapest": (47.4979, 19.0402), "bucharest": (44.4268, 26.1025),
    "cluj-napoca": (46.7712, 23.6236), "cluj": (46.7712, 23.6236), "timisoara": (45.7489, 21.2087), "iasi": (47.1585, 27.6014), 
    "brasov": (45.6427, 25.5887), "sofia": (42.6977, 23.3219), "belgrade": (44.7866, 20.4489), "zagreb": (45.8150, 15.9819),
    "ljubljana": (46.0569, 14.5058), "bratislava": (48.1486, 17.1077), "tallinn": (59.4370, 24.7536), "riga": (56.9496, 24.1052),
    "vilnius": (54.6872, 25.2797), "kyiv": (50.4501, 30.5234), "kiev": (50.4501, 30.5234), "istanbul": (41.0082, 28.9784),
    "ankara": (39.9334, 32.8597), "athens": (37.9838, 23.7275), "moscow": (55.7558, 37.6173), "saint petersburg": (59.9311, 30.3609),

    # Middle East
    "dubai": (25.2048, 55.2708), "abu dhabi": (24.4539, 54.3773), "riyadh": (24.7136, 46.6753), "jeddah": (21.4858, 39.1925),
    "doha": (25.2854, 51.5310), "tel aviv": (32.0853, 34.7818), "jerusalem": (31.7683, 35.2137), "beirut": (33.8938, 35.5018),
    "amman": (31.9454, 35.9284), "cairo": (30.0444, 31.2357),

    # Asia — East
    "tokyo": (35.6762, 139.6503), "osaka": (34.6937, 135.5023), "yokohama": (35.4437, 139.6380), "nagoya": (35.1815, 136.9066),
    "kyoto": (35.0116, 135.7681), "fukuoka": (33.5904, 130.4017), "beijing": (39.9042, 116.4074), "shanghai": (31.2304, 121.4737),
    "shenzhen": (22.5431, 114.0579), "guangzhou": (23.1291, 113.2644), "chengdu": (30.5728, 104.0668), "hangzhou": (30.2741, 120.1551),
    "hong kong": (22.3193, 114.1694), "taipei": (25.0330, 121.5654), "seoul": (37.5665, 126.9780), "busan": (35.1796, 129.0756),

    # Asia — Southeast
    "singapore": (1.3521, 103.8198), "bangkok": (13.7563, 100.5018), "chiang mai": (18.7883, 98.9853),"kuala lumpur": (3.1390, 101.6869),
    "penang": (5.4164, 100.3327), "jakarta": (-6.2088, 106.8456), "manila": (14.5995, 120.9842), "ho chi minh city": (10.8231, 106.6297),
    "hanoi": (21.0278, 105.8342),

    # Asia — South
    "mumbai": (19.0760, 72.8777), "delhi": (28.7041, 77.1025), "new delhi": (28.6139, 77.2090), "bangalore": (12.9716, 77.5946),
    "bengaluru": (12.9716, 77.5946), "hyderabad": (17.3850, 78.4867), "chennai": (13.0827, 80.2707), "pune": (18.5204, 73.8567),
    "kolkata": (22.5726, 88.3639), "ahmedabad": (23.0225, 72.5714),

    # Oceania
    "sydney": (-33.8688, 151.2093), "melbourne": (-37.8136, 144.9631), "brisbane": (-27.4698, 153.0251), "perth": (-31.9505, 115.8605),
    "auckland": (-36.8485, 174.7633), "wellington": (-41.2865, 174.7762),

    # South America
    "são paulo": (-23.5505, -46.6333), "sao paulo": (-23.5505, -46.6333), "rio de janeiro": (-22.9068, -43.1729), "buenos aires": (-34.6037, -58.3816),
    "bogota": (4.7110, -74.0721), "lima": (-12.0464, -77.0428), "santiago": (-33.4489, -70.6693), "medellin": (6.2442, -75.5812),
    "montevideo": (-34.9011, -56.1645),

    # Africa
    "cape town": (-33.9249, 18.4241), "johannesburg": (-26.2041, 28.0473), "lagos": (6.5244, 3.3792), "nairobi": (-1.2921, 36.8219),
    "casablanca": (33.5731, -7.5898), "accra": (5.6037, -0.1870), "addis ababa": (9.0250, 38.7469), "dar es salaam": (-6.7924, 39.2083),
}


def geocode(location: str) -> Optional[tuple[float, float]]:

    if not location: 
        return None

    loc = location.lower().strip()

    for suffix in [" area", " region", " metro", " metropolitan"]:
        if loc.endswith(suffix):
            loc = loc[:-len(suffix)].strip()

    # 1. Exact match
    if loc in CITIES: return CITIES[loc]

    # 2. Try without country qualifiers ("Boston, US" → "boston")
    base = re.split(r"[,/]", loc)[0].strip()
    if base in CITIES: return CITIES[base]

    # 3. Partial match — find cities that start with the query
    for city_name, coords in CITIES.items():
        if city_name.startswith(loc) or loc.startswith(city_name):
            return coords

    return None


def infer_radius_km(modifier: str) -> float:

    if not modifier:
        return DEFAULT_RADIUS_KM
    modifier = modifier.lower().strip()

    km_match = re.search(r"(\d+)\s*(?:km|kilometers?|kilometres?)", modifier)
    if km_match:
        return float(km_match.group(1))

    mi_match = re.search(r"(\d+)\s*(?:mi|miles?)", modifier)
    if mi_match:
        return float(mi_match.group(1)) * 1.60934

    if any(w in modifier for w in ["very close", "right next", "walking distance"]):
        return 15
    if any(w in modifier for w in ["near", "nearby", "close to", "neighborhood"]):
        return 50
    if any(w in modifier for w in ["not far", "not too far", "short distance", "driving distance"]):
        return 100
    if any(w in modifier for w in ["greater area", "region of", "surrounding", "outskirts"]):
        return 75
    if any(w in modifier for w in ["around", "area of", "vicinity"]):
        return 60

    return DEFAULT_RADIUS_KM