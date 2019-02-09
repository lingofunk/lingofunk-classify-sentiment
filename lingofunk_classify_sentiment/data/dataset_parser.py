import json

CITY = 'Brooklyn'
business_input = open('yelp_academic_dataset_business.json', 'r', encoding='utf-8')
reviews_input = open('yelp_academic_dataset_review.json', 'r', encoding='utf-8')
output = open(CITY + '_dataset.json', 'w', encoding='utf-8')

if __name__ == '__main__':
    business_ids = set()
    row_str = business_input.readline()
    output.write(r'{"data":[')
    while row_str != '':
        json_str = json.JSONDecoder().decode(row_str)
        if json_str['city'] == CITY:
            business_ids.add(json_str['business_id'])
        row_str = business_input.readline()
    row_str = reviews_input.readline()
    output_str = ''
    while row_str != '':
        json_str = json.JSONDecoder().decode(row_str)
        if json_str['business_id'] in business_ids:
            if output_str != '':
                output.write(output_str[:-1] + ',')
            output_str = row_str
        row_str = reviews_input.readline()
    output.write(output_str + r']}')
    output.close()
