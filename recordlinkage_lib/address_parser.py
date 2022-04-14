import numpy as np
import pandas as pd
import os
import re
from postal.parser import parse_address # install: https://github.com/openvenues/libpostal
from utils.settings import get_project_root

ROOT = get_project_root()


def parse_address_fields(address=None):
    dict_address = {
        'street_number': np.nan,
        'street_type': np.nan,
        'street_name': np.nan,
        'address_line2': np.nan,
        'postal_code': np.nan,
        'city': np.nan
    }

    if type(address) == float:
        pass

    else:
        addline2 = ''
        parsed_add = parse_address(address)

        for item in parsed_add:

            if item[1] == 'city':
                dict_address['city'] = item[0]

            elif item[1] == 'postcode':
                try:
                    dict_address['postal_code'] = int(item[0])
                except ValueError:
                    dict_address['postal_code'] = item[0]

            elif item[1] == 'road':
                road_comps = item[0].split()
                dict_address['street_name'] = ' '.join(road_comps[1:])
                if road_comps[0] == 'r':
                    dict_address['street_type'] = 'rue'
                elif road_comps[0] in unique_street_types:
                    dict_address['street_type'] = road_comps[0]
                else:
                    dict_address['street_name'] = item[0]

            elif item[1] == 'house_number':
                try:
                    dict_address['street_number'] = int(item[0])
                except ValueError:
                    # print('The parsed \'house_number\' is not a valid integer. Extracting the first integer '\
                    #     'occurrence (from string) and setting it as \'street_number\'')
                    ints = re.findall(r'\d+', item[0])
                    if len(ints) == 0:
                        dict_address['street_number'] = np.nan
                    else:
                        sep_ints = [int(s) for s in item[0].split() if s.isdigit()]
                        if len(sep_ints) != 0:
                            dict_address['street_number'] = sep_ints[0]
                        else:
                            dict_address['street_number'] = ints[0]

            else:
                addline2 += item[0]
                addline2 += ' '
        if addline2 == '':
            dict_address['address_line2'] = np.nan
        else:
            dict_address['address_line2'] = addline2

    return dict_address


if __name__ == '__main__':
    s1_cstr = pd.read_csv(os.path.join(ROOT, 'data' 'source1_cstr.csv'))
    s2_nond_cstr = pd.read_csv(os.path.join(ROOT, 'data', 'source2_cstr.csv'))
    unique_street_types = np.unique(s1_cstr['street_type'].dropna())

    s2_nond_cstr[['street_number', 'street_type', 'street_name', 'address_line2', 'postal_code', 'city']] = \
        s2_nond_cstr.apply(lambda row: parse_address_fields(row.address), axis=1, result_type='expand')

    s2_cstr_parsed_address = s2_nond_cstr.drop('address', axis=1)
    s2_cstr_parsed_address.to_csv(os.path.join(ROOT, 'data', 'source2_cstr_parsedaddress.csv'), index=False)



