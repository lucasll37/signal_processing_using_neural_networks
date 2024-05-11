import requests as req
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from .mapHandler import GeoTIFFHandler

from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime


# Define a function to add a gap of specified size and color to an image
# This function is commented out and not used in the code
# def add_gap(image, gap_size, gap_color=[255, 255, 255]):
#     return np.vstack(
#         (image, np.full((gap_size, image.shape[1], 3), gap_color, dtype=np.uint8))
#     )


# Initialize an empty list to store flight itineraries
itineraries = []

# Define coordinates for the airports
coordinates = {
    "SBGR": [-23.4323, -46.4695],
    "SBCF": [-19.6357, -43.9669],
    "SBRJ": [-22.9104, -43.1632],
    "SBPA": [-29.9949, -51.1763],
    "SBSV": [-12.9162, -38.3342],
    "SBFL": [-27.6745, -48.5462],
    "SBRF": [-8.1287,  -34.9259],
    "SBBR": [-15.8700, -47.9210],
    "SBCT": [-25.5299, -49.1724],
    "SBSP": [-23.6282, -46.6570],
    "SBKP": [-23.0069, -47.1344],
    "SBGL": [-22.8145, -43.2466],
}

# Define locations with their corresponding coordinates
# location = {
#     "SBGR": [1665, 1459],
#     "SBCF": [1722, 1376],
#     "SBRJ": [1739, 1447],
#     "SBPA": [1557, 1614],
#     "SBSV": [1853, 1217],
#     "SBFL": [1618, 1562],
#     "SBRF": [1932, 1106],
#     "SBBR": [1632, 1286],
#     "SBCT": [1599, 1509],
#     "SBSP": [1662, 1469],
#     "SBKP": [1648, 1453],
#     "SBGL": [1740, 1446],
# }

map_handler = GeoTIFFHandler("../../data/map_geo.tif")
location = map_handler.convert_airport_coordinates(coordinates)

# Generate all possible flight itineraries
for aero_from in location.keys():
    for aero_to in location.keys():
        if aero_from == aero_to:
            continue

        itineraries.append(f"{aero_from}_{aero_to}")

# Custom transformer class for handling satellite images and extracting features
class SatelliteImageHandler(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        width=20,
        outputWidth=32,
        outputHeight=32,
        printRoutes=False,
        printEachImage=False,
    ):
        self.width = width
        self.outputWidth = outputWidth
        self.outputHeight = outputHeight
        self.printRoutes = printRoutes
        self.printEachImage = printEachImage

    def fit(self, X, y=None):
        return self

    # Function to add a gap to an image (not used in the code)
    def _add_gap(self, image, gap_size, gap_color=[255, 255, 255]):
        return np.vstack(
            (image, np.full((gap_size, image.shape[1], 3), gap_color, dtype=np.uint8))
        )

    # Main function to transform input data
    def transform(self, X, y=None):
        print(f'Started SatelliteImageHandler: {datetime.now().strftime("%H:%M:%SZ")}')

        _X = X.copy() # Make a copy of input data

        route = []  # Store flight routes
        hora_ref = []  # Store time references
        distance = []  # Store distances between airports

        imageSatelite_red = []  # Store mean red intensity in satellite images
        imageSatelite_yellow = []  # Store mean yellow intensity
        imageSatelite_green = []  # Store mean green intensity
        imageSatelite_blue = []  # Store mean blue intensity

        firstCicle = True # Flag for the first iteration

        # Drop duplicate rows based on "hora_ref" and "url_img_satelite" columns
        df = X[["hora_ref", "url_img_satelite"]].drop_duplicates().copy()

        # Iterate over unique rows of "hora_ref" and "url_img_satelite"
        for _, row in df.iterrows():

            # Attempt to retrieve the image from the URL
            while True:
                try:
                    response = req.get(row["url_img_satelite"]).content
                    break

                except:
                    break

            # Convert retrieved image to OpenCV BGR format and then to RGB
            arrayImage = np.asarray(bytearray(response), dtype=np.uint8)
            cv2ImageBGR = cv2.imdecode(arrayImage, cv2.IMREAD_COLOR)
            cv2ImageRGB = cv2.cvtColor(cv2ImageBGR, cv2.COLOR_BGR2RGB)

            if firstCicle:
                cv2ImageRGBCopy = cv2ImageRGB.copy()

            # Iterate over all possible flight itineraries
            for itinerary in itineraries:
                [origin, destiny] = itinerary.split("_")
                p_from = np.array(location[origin])
                p_to = np.array(location[destiny])

                # Calculate vectors and points for perspective transformation
                # based on the flight route
                v = p_to - p_from
                v = v / np.linalg.norm(v)
                v_perp = np.array([v[1], -v[0]])

                padding = self.width / 2
                p2 = p_to + v * padding + v_perp * padding
                p3 = p_to + v * padding - v_perp * padding
                p1 = p_to - v * padding + v_perp * padding
                p4 = p_to - v * padding - v_perp * padding

                # p1 = p_from - v * padding + v_perp * padding
                # p4 = p_from - v * padding - v_perp * padding

                if firstCicle:
                    pts = np.array([p1, p2, p3, p4], np.int32)
                    pts = pts.reshape((-1, 1, 2))

                    color = (255, 0, 0)
                    thickness = 2
                    cv2.polylines(
                        cv2ImageRGBCopy,
                        [pts],
                        isClosed=True,
                        color=color,
                        thickness=thickness,
                    )

                satelitePoints = np.array([p1, p2, p3, p4], dtype=np.float32)
                outputPoints = np.array(
                    [
                        [0, 0],
                        [self.outputWidth, 0],
                        [self.outputWidth, self.outputHeight],
                        [0, self.outputHeight],
                    ],
                    dtype=np.float32,
                )

                matriz = cv2.getPerspectiveTransform(satelitePoints, outputPoints)

                outputImage = cv2.warpPerspective(
                    cv2ImageRGB, matriz, (self.outputWidth, self.outputHeight)
                )

                arrayImageOutput = np.array(outputImage)

                hsv = cv2.cvtColor(arrayImageOutput, cv2.COLOR_RGB2HSV)

                # Define color ranges for each color
                red_lower_limit1 = np.array([0, 50, 50])
                red_upper_limit1 = np.array([12, 255, 255])
                red_lower_limit2 = np.array([150, 50, 50])
                red_upper_limit2 = np.array([180, 255, 255])

                yellow_lower_limit1 = np.array([22, 50, 50])
                yellow_upper_limit1 = np.array([30, 255, 255])
                yellow_lower_limit2 = np.array([31, 50, 50])
                yellow_upper_limit2 = np.array([38, 255, 255])

                green_lower_limit1 = np.array([40, 50, 50])
                green_upper_limit1 = np.array([70, 255, 255])
                green_lower_limit2 = np.array([71, 50, 50])
                green_upper_limit2 = np.array([80, 255, 255])

                blue_lower_limit1 = np.array([100, 50, 50])
                blue_upper_limit1 = np.array([120, 255, 255])
                blue_lower_limit2 = np.array([121, 50, 50])
                blue_upper_limit2 = np.array([140, 255, 255])

                # Create masks for each color range
                mask_red1 = cv2.inRange(hsv, red_lower_limit1, red_upper_limit1)
                mask_red2 = cv2.inRange(hsv, red_lower_limit2, red_upper_limit2)
                mask_red = cv2.bitwise_or(mask_red1, mask_red2)
                outputImage_red = cv2.bitwise_and(
                    arrayImageOutput, arrayImageOutput, mask=mask_red
                )
                mean_r = outputImage_red.mean()

                mask_yellow1 = cv2.inRange(hsv, yellow_lower_limit1, yellow_upper_limit1)
                mask_yellow2 = cv2.inRange(hsv, yellow_lower_limit2, yellow_upper_limit2)
                mask_yellow = cv2.bitwise_or(mask_yellow1, mask_yellow2)
                outputImage_yellow = cv2.bitwise_and(
                    arrayImageOutput, arrayImageOutput, mask=mask_yellow
                )
                mean_y = outputImage_yellow.mean()

                mask_green1 = cv2.inRange(hsv, green_lower_limit1, green_upper_limit1)
                mask_green2 = cv2.inRange(hsv, green_lower_limit2, green_upper_limit2)
                mask_green = cv2.bitwise_or(mask_green1, mask_green2)
                outputImage_green = cv2.bitwise_and(
                    arrayImageOutput, arrayImageOutput, mask=mask_green
                )
                mean_g = outputImage_green.mean()

                mask_blue1 = cv2.inRange(hsv, blue_lower_limit1, blue_upper_limit1)
                mask_blue2 = cv2.inRange(hsv, blue_lower_limit2, blue_upper_limit2)
                mask_blue = cv2.bitwise_or(mask_blue1, mask_blue2)
                outputImage_blue = cv2.bitwise_and(
                    arrayImageOutput, arrayImageOutput, mask=mask_blue
                )
                mean_b = outputImage_blue.mean()

                # Append extracted features to respective lists
                route.append(itinerary)
                hora_ref.append(row["hora_ref"])
                distance.append(np.linalg.norm(p_to - p_from))
                imageSatelite_red.append(mean_r)
                imageSatelite_yellow.append(mean_y)
                imageSatelite_green.append(mean_g)
                imageSatelite_blue.append(mean_b)

                # Optionally, display each transformed image
                if self.printEachImage:
                    gap_size = 1
                    arrayImageOutput_with_gap = self._add_gap(
                        arrayImageOutput, gap_size
                    )
                    outputImage_red_with_gap = self._add_gap(outputImage_red, gap_size)
                    outputImage_yellow_with_gap = self._add_gap(
                        outputImage_yellow, gap_size
                    )
                    outputImage_green_with_gap = self._add_gap(
                        outputImage_green, gap_size
                    )

                    concatenated_image = np.vstack(
                        (
                            arrayImageOutput_with_gap,
                            outputImage_red_with_gap,
                            outputImage_yellow_with_gap,
                            outputImage_green_with_gap,
                            outputImage_blue,
                        )
                    )

                    # plt.figure(figsize=(10, 10))
                    # plt.figure(figsize=(10, 10))
                    plt.imshow(concatenated_image)
                    plt.title(
                        f"{itinerary} - {row['hora_ref']} r: {mean_r: .1f} y: {mean_y: .1f} g: {mean_g: .1f} b: {mean_b: .1f}",
                        fontsize=6,
                    )
                    plt.axis("off")
                    plt.show()

            # Optionally, display the flight routes on the satellite image
            if firstCicle and self.printRoutes:
                plt.figure(figsize=(8, 8))
                plt.imshow(cv2ImageRGBCopy[800:1700, 1000:-200])
                plt.show()
                plt.close()

                firstCicle = False

        # Create a DataFrame to store the extracted features
        imagesRouteData = pd.DataFrame(
            {
                "route": route,
                "hora_ref": hora_ref,
                "distance": distance,
                "imageSatelite_red": imageSatelite_red,
                "imageSatelite_yellow": imageSatelite_yellow,
                "imageSatelite_green": imageSatelite_green,
                "imageSatelite_blue": imageSatelite_blue,
            }
        )

        cols = [
            "distance",
            "imageSatelite_red",
            "imageSatelite_yellow",
            "imageSatelite_green",
            "imageSatelite_blue",
        ]

        # Merge the DataFrame with the original input data
        _X["route"] = _X["origem"] + "_" + _X["destino"]
        _X.reset_index(inplace=True, drop=False)
        _X = _X.merge(imagesRouteData, on=["hora_ref", "route"], how="left")
        _X.set_index("flightid", inplace=True)

        # Fill missing values and drop unnecessary columns
        _X[cols] = _X[cols].fillna(0)
        _X.drop(["route"], axis=1, inplace=True)

        # Print a message indicating the completion of the transformation process
        print(f'Finished SatelliteImageHandler: {datetime.now().strftime("%H:%M:%SZ")}')

        return _X
