from osgeo import gdal, osr

class GeoTIFFHandler:
    def __init__(self, file_path):
        self.dataset = gdal.Open(file_path)
        self.gt = self.dataset.GetGeoTransform()
        self.source = osr.SpatialReference()
        self.source.ImportFromEPSG(4326)  # WGS84

        self.target = osr.SpatialReference()
        self.target.ImportFromWkt(self.dataset.GetProjection())

        self.transform = osr.CoordinateTransformation(self.source, self.target)

    def latlon_to_pixel(self, lat, lon):
        lon, lat, _ = self.transform.TransformPoint(lon, lat)
        x = int((lon - self.gt[0]) / self.gt[1])
        y = int((lat - self.gt[3]) / self.gt[5])
        return x, y

    def convert_airport_coordinates(self, airport_coordinates):
        airport_pixel_coordinates = {}
        for airport, coordinates in airport_coordinates.items():
            pixel_x, pixel_y = self.latlon_to_pixel(coordinates[0], coordinates[1])
            airport_pixel_coordinates[airport] = (pixel_x, pixel_y)
        return airport_pixel_coordinates

# Example usage:
if __name__ == "__main__":
    file_path = '../../data/map_geo.tif'
    airport_coordinates = {
        "SBGR": [-23.4323, -46.4695],
        "SBCF": [-19.6357, -43.9669],
        "SBRJ": [-22.9104, -43.1632],
        "SBPA": [-29.9949, -51.1763],
        "SBSV": [-12.9162, -38.3342],
        "SBFL": [-27.6745, -48.5462],
        "SBRF": [-8.1287, -34.9259],
        "SBBR": [-15.8700, -47.9210],
        "SBCT": [-25.5299, -49.1724],
        "SBSP": [-23.6282, -46.6570],
        "SBKP": [-23.0069, -47.1344],
        "SBGL": [-22.8145, -43.2466],
    }

    geo_handler = GeoTIFFHandler(file_path)
    airport_pixel_coordinates = geo_handler.convert_airport_coordinates(airport_coordinates)

    print(airport_coordinates)
