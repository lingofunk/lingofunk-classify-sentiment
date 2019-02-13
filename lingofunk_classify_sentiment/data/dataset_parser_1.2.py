import json
import pandas as pd
import geojson

define extract_top_business(city, type="Restaurant")
CITY = "Toronto"
business_input = open("yelp_academic_dataset_business.json", "r", encoding="utf-8")
reviews_input = open("yelp_academic_dataset_review.json", "r", encoding="utf-8")
ouput_geojson = open(CITY + "_dataset.geojson", "w", encoding="utf-8")
CITY_NUMBER = 100
REVIEW_PER_BUSINESS = 10


if __name__ == "__main__":
    business_ids = set()
    business = dict()
    business_rate = dict()
    business_reviews = dict()

    row_str = business_input.readline()
    while row_str != "":
        json_str = json.JSONDecoder().decode(row_str)
       if json_str["city"] == CITY:
            # if json_str['state'] == STATE:
            business_ids.add(json_str["business_id"])
            business_rate.update({json_str["business_id"]: json_str["review_count"]})
            business.update(
                {
                    json_str["business_id"]: {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [
                                json_str["longitude"],
                                json_str["latitude"],
                            ],
                        },
                        "properties": dict().fromkeys(
                            [
                                i
                                for i in json_str.keys()
                                if i not in ["latitude", "longitude"]
                            ]
                        ),
                    }
                }
            )
            for i in json_str.keys():
                if i in business[json_str["business_id"]]["properties"].keys():
                    business[json_str["business_id"]]["properties"][i] = json_str[i]
            business[json_str["business_id"]]["properties"].update({"reviews": []})
        row_str = business_input.readline()
    ans = []
    business_rate = sorted(business_rate.items(), key=lambda x: -x[1])
    business_ids = set()
    for i in range(CITY_NUMBER):
        business_ids.add(business_rate[i][0])
        business_reviews.update({business_rate[i][0]: list()})
    business = dict([(i, business[i]) for i in business_ids])
    row_str = reviews_input.readline()
    r_count = 0
    while row_str != "":
        json_str = json.JSONDecoder().decode(row_str)
        if json_str["business_id"] in business_ids:
            if len(business_reviews[json_str["business_id"]]) < REVIEW_PER_BUSINESS:
                business_reviews[json_str["business_id"]].append(json_str)
                r_count += 1
            else:
                business_reviews[json_str["business_id"]] = sorted(
                    business_reviews[json_str["business_id"]],
                    key=lambda x: -(x["useful"] + x["funny"] + x["cool"]),
                )
                if (
                    json_str["useful"] + json_str["funny"] + json_str["cool"]
                    > business_reviews[json_str["business_id"]][
                        REVIEW_PER_BUSINESS - 1
                    ]["useful"]
                    + business_reviews[json_str["business_id"]][
                        REVIEW_PER_BUSINESS - 1
                    ]["funny"]
                    + business_reviews[json_str["business_id"]][
                        REVIEW_PER_BUSINESS - 1
                    ]["cool"]
                ):
                    business_reviews[json_str["business_id"]][
                        REVIEW_PER_BUSINESS - 1
                    ] = json_str
        row_str = reviews_input.readline()
    reviews_to_out = list()
    for i in business_reviews.keys():
        business[i]["properties"]["reviews"] = sorted(
            business_reviews[i], key=lambda x: -(x["useful"] + x["funny"] + x["cool"])
        )
        reviews_to_out.extend(business[i]["properties"]["reviews"])
    business_to_out = geojson.GeoJSONEncoder().encode(
        {"type": "FeatureCollection", "features": list(business.values())}
    )
    pd.DataFrame(reviews_to_out).to_csv(
        CITY + "_dataset.csv", index=False, encoding="utf-8"
    )
    ouput_geojson.write(business_to_out)
    ouput_geojson.close()
