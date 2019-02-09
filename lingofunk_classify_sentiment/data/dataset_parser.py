import json

CITY = 'Toronto'
business_input = open('business.json', 'r', encoding='utf-8')
reviews_input = open('dataset.json', 'r', encoding='utf-8')
output = open(CITY + '_dataset.json', 'w', encoding='utf-8')

if __name__ == '__main__':
    business_ids = set()
    str = business_input.readline()
    while str != '':
        json_str = json.JSONDecoder().decode(str)
        if json_str['city'] == CITY:
            business_ids.add(json_str['business_id'])
        str = business_input.readline()
    str = reviews_input.readline()
    while str != '':
        json_str = json.JSONDecoder().decode(str)
        if json_str['business_id'] in business_ids:
            output.write(str)
        str = reviews_input.readline()
    output.close()
