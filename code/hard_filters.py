import logging
import math
from code.taxonomy import COUNTRY_ALIASES, ISO2_TO_COUNTRY
from code.models import CompanyProfile, HardFilters, ParsedQuery
logger = logging.getLogger(__name__)

MIN_HARD_FILTER_RESULTS = 7


def _compute_distance( latitude1:float, latitude2: float, longitude1: float, longitude2: float ) -> float:

    R = 6371.0
    dist_lat = math.radians(latitude2-latitude1)
    dist_long = math.radians(longitude2-longitude1)
    a = ( math.sin(dist_lat / 2) ** 2 + math.cos(math.radians(latitude1)) * math.cos(math.radians(latitude2)) * math.sin(dist_long / 2) ** 2 )

    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def apply_hard_filters(companies: list[CompanyProfile], parsed_query: ParsedQuery)-> list[CompanyProfile]:

    filters = parsed_query.hard_filters
    trace = parsed_query.trace
    results = []

    active = []
    if filters.locations: active.append(f"location ({len(filters.locations)} targets)")
    if filters.excluded_locations: active.append(f"NOT location ({len(filters.excluded_locations)} excluded)")
    if filters.employee_count_min: active.append(f"employee >= {filters.employee_count_min}")
    if filters.employee_count_max: active.append(f"employee <= {filters.employee_count_max}")
    if filters.revenue_min: active.append(f"revenue >= {filters.revenue_min:,.0f}")
    if filters.revenue_max: active.append(f"revenue <= {filters.revenue_max:,.0f}")
    if filters.year_founded_min: active.append(f"year >= {filters.year_founded_min}")
    if filters.year_founded_max: active.append(f"year <= {filters.year_founded_max}")
    if filters.is_public is not None: active.append(f"public = {filters.is_public}")
    if filters.naics_codes: active.append(f"NAICS ({len(filters.naics_codes)} prefixes)")
    if filters.excluded_naics_codes: active.append(f"NOT NAICS ({len(filters.excluded_naics_codes)} excluded)")
    if filters.geo_lat is not None: active.append(f"geolocation ({filters.geo_radius_km:.0f} km from {filters.geo_lat:.2f},{filters.geo_lng:.2f})")
    trace.add("stage_1", "active_filters", ", ".join(active) if active else "none (passthrough)")

    for company in companies:
        if _passes_all_filters(company, filters):
            results.append(company)
    trace.add("stage_1", "initial_result", f"{len(companies)} -> {len(results)}")

    # If NAICS filter was active and eliminated too aggressively, relax it
    min_threshold = max(3, int(len(companies) * 0.01))
    if len(results) < min_threshold and filters.naics_codes:
        relaxed_filters = HardFilters( 
            locations=filters.locations, excluded_locations=filters.excluded_locations,
            year_founded_min=filters.year_founded_min, year_founded_max=filters.year_founded_max,
            employee_count_min=filters.employee_count_min, employee_count_max=filters.employee_count_max, revenue_min=filters.revenue_min,
            revenue_max=filters.revenue_max, is_public=filters.is_public, naics_codes=None,
            excluded_naics_codes=filters.excluded_naics_codes, 
            geo_lat=filters.geo_lat, geo_lng=filters.geo_lng,
            geo_radius_km=filters.geo_radius_km,
        )

        relaxed_results = []
        for company in companies:
            if _passes_all_filters(company, relaxed_filters):
                relaxed_results.append(company)

        if len(relaxed_results) > len(results):
            trace.add("stage_1", "naics_relaxed", f"{len(results)} too few -> relaxed to {len(relaxed_results)} (NAICS moved to soft scoring)")
            results = relaxed_results
            parsed_query.naics_was_relaxed = True

    trace.add("stage_1", "final_result", f"{len(results)} companies passed")
    logger.info(f"Hard filters: {len(companies)} -> {len(results)} companies ({len(companies) - len(results)} eliminated)" )

    return results


def _passes_all_filters(company: CompanyProfile, filters: HardFilters) -> bool:

    # Geo proximity filter (takes priority over country-level location filter)
    if filters.geo_lat is not None and filters.geo_lng is not None:
        if company.latitude is not None and company.longitude is not None:
            dist = _compute_distance( filters.geo_lat, company.latitude, filters.geo_lng, company.longitude)
            if dist > filters.geo_radius_km:
                return False
        else:
            return False
    elif filters.locations:
        if not _check_location(company, filters.locations):
            return False

    if filters.year_founded_min is not None:
        if company.year_founded is not None:
            if company.year_founded < filters.year_founded_min:
                return False

    if filters.year_founded_max is not None:
        if company.year_founded is not None:
            if company.year_founded > filters.year_founded_max:
                return False

    if filters.employee_count_min is not None:
        if company.employee_count is not None:
            if company.employee_count < filters.employee_count_min:
                return False

    if filters.employee_count_max is not None:
        if company.employee_count is not None:
            if company.employee_count > filters.employee_count_max:
                return False
            
    if filters.revenue_min is not None:
        if company.revenue is not None:
            if company.revenue < filters.revenue_min:
                return False

    if filters.revenue_max is not None:
        if company.revenue is not None:
            if company.revenue > filters.revenue_max:
                return False
            
    if filters.is_public is not None:
        if company.is_public is not None:
            if company.is_public != filters.is_public:
                return False

    # NAICS code filter (prefix matching)
    if filters.naics_codes:
        if not _check_naics(company, filters.naics_codes):
            return False

    if filters.excluded_naics_codes:
        if _check_naics(company, filters.excluded_naics_codes):
            # Company MATCHES an excluded industry -> reject
            return False

    if filters.excluded_locations:
        if _check_location(company, filters.excluded_locations):
            # Company IS in an excluded location -> reject
            return False

    return True


def _check_location(company: CompanyProfile, locations: list[str]) -> bool:
    """
    If a company matches any of the target locations, we're using ISO2_TO_COUNTRY for country_code resolution that bypasses ambiguity filter.
    Falls back to address/region substring match.
        > If company has no location data, it will pass due to missing data tolerance.
    """

    if company.country_code:
        company_country_code = company.country_code.lower()
        
        # Resolve ISO2 code -> canonical country name
        # ISO2_TO_COUNTRY has ALL codes including "us","no","in","it" etc.
        canonical_cc = ISO2_TO_COUNTRY.get(company_country_code) or COUNTRY_ALIASES.get(company_country_code, company_country_code)

        for target_location in locations:
            normalized_location = target_location.lower()
            
            # Direct code match: filter="us" matches country_code="us"
            if normalized_location == company_country_code:
                return True

            # Canonical match: filter="united states" matches canonical("us")="united states"
            canonical_loc = COUNTRY_ALIASES.get(normalized_location, normalized_location)
            if canonical_cc == canonical_loc:
                return True

    searchable_parts = []
    if company.address:
        searchable_parts.append(company.address.lower())
    if company.region_name:
        searchable_parts.append(company.region_name.lower())
    if company.town:
        searchable_parts.append(company.town.lower())

    if not searchable_parts and not company.country_code:
        return True 

    combined = " ".join(searchable_parts)
    return any(loc.lower() in combined for loc in locations)


def _check_naics(company: CompanyProfile, naics_prefixes: list[str]) -> bool:

    company_codes = company.get_all_naics_codes()
    if not company_codes:
        return True 
    
    for company_code in company_codes:
        for prefix in naics_prefixes:
            if company_code.startswith(prefix):
                return True

    return False