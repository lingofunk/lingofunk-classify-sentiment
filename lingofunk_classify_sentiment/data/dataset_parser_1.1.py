import json
import pandas as pd

CITY = 'Toronto'
business_input = open('yelp_academic_dataset_business.json', 'r', encoding='utf-8')
reviews_input = open('yelp_academic_dataset_review.json', 'r', encoding='utf-8')
#output_csv = open(CITY + '_dataset.csv', 'w', encoding='utf-8')
CITY_NUMBER = 100

def convert(x):
    ''' Convert a json string to a flat python dictionary
    which can be passed into Pandas. '''
    ob = json.loads(x)
    for k, v in ob.items():
        if isinstance(v, list):
            ob[k] = ','.join(v)
        elif isinstance(v, dict):
            for kk, vv in v.items():
                ob['%s_%s' % (k, kk)] = vv
            del ob[k]
    return ob


if __name__ == '__main__':
    business_ids = set()
    business_rate = dict()
    row_str = business_input.readline().replace('\n', ' ')
    while row_str != '':
        json_str = json.JSONDecoder().decode(row_str)
        if json_str['city'] == CITY:
            business_ids.add(json_str['business_id'])
            business_rate.update({json_str['business_id']: [0, 0]})
        row_str = business_input.readline().replace('\n', ' ')
    row_str = reviews_input.readline().replace('\n', ' ')
    ans = []
    while row_str != '':
        json_str = json.JSONDecoder().decode(row_str)
        if json_str['business_id'] in business_ids:
            business_rate[json_str['business_id']][0] += json_str['stars']
            business_rate[json_str['business_id']][1] += 1
        row_str = reviews_input.readline().replace('\n', ' ')
    reviews_input.close()
    business_rate = (sorted(business_rate.items(), key=lambda x: -x[1][1]))
    business_ids = set()
    s = 0
    for i in range(CITY_NUMBER):
        business_ids.add(business_rate[i][0])
        s += business_rate[i][1][1]
    reviews_input = open('yelp_academic_dataset_review.json', 'r', encoding='utf-8')
    row_str = reviews_input.readline().replace('\n', ' ')
    while row_str != '':
        json_str = json.JSONDecoder().decode(row_str)
        if json_str['business_id'] in business_ids:
            ans.append(convert(row_str))
        row_str = reviews_input.readline().replace('\n', ' ')
    a = pd.DataFrame(ans)
    a.to_csv(CITY + '_dataset.csv', encoding='utf-8', index=False)
    print('!', s, '!')
