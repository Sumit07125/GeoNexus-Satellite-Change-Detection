// WGS84 local-scale estimates for compact Maharashtra AOIs. These are used
// consistently for UI labels and the center-based square-box controls.
export function bboxDimensionsMeters(bbox) {
  if (!bbox) return null;
  const latMid = ((Number(bbox.lat_min) + Number(bbox.lat_max)) / 2) * Math.PI / 180;
  const latSpan = Math.abs(Number(bbox.lat_max) - Number(bbox.lat_min));
  const lonSpan = Math.abs(Number(bbox.lon_max) - Number(bbox.lon_min));
  const metersPerLatitudeDegree = 111132.92 - 559.82 * Math.cos(2 * latMid) + 1.175 * Math.cos(4 * latMid) - 0.0023 * Math.cos(6 * latMid);
  const metersPerLongitudeDegree = 111412.84 * Math.cos(latMid) - 93.5 * Math.cos(3 * latMid) + 0.118 * Math.cos(5 * latMid);
  return {
    width: lonSpan * metersPerLongitudeDegree,
    height: latSpan * metersPerLatitudeDegree,
  };
}

export function bboxDimensionsKm(bbox) {
  const meters = bboxDimensionsMeters(bbox);
  if (!meters) return null;
  const widthKm = meters.width / 1000;
  const heightKm = meters.height / 1000;
  return {
    widthKm,
    heightKm,
    areaKm2: widthKm * heightKm,
    isSubModelMinimum: meters.width < 1280 || meters.height < 1280,
  };
}

export function createSquareBbox(latitude, longitude, sizeKm) {
  const lat = Number(latitude);
  const lon = Number(longitude);
  const sizeM = Number(sizeKm) * 1000;
  const centerLat = lat * Math.PI / 180;
  const halfLatDeg = (sizeM / 2) / (111132.92 - 559.82 * Math.cos(2 * centerLat) + 1.175 * Math.cos(4 * centerLat) - 0.0023 * Math.cos(6 * centerLat));
  const metersPerLonDegree = 111412.84 * Math.cos(centerLat) - 93.5 * Math.cos(3 * centerLat) + 0.118 * Math.cos(5 * centerLat);
  const halfLonDeg = (sizeM / 2) / Math.max(1, metersPerLonDegree);
  return {
    lon_min: lon - halfLonDeg,
    lat_min: lat - halfLatDeg,
    lon_max: lon + halfLonDeg,
    lat_max: lat + halfLatDeg,
  };
}
