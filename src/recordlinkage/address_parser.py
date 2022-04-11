import numpy as np
import pandas as pd
import os
import re
from postal.parser import parse_address # install: https://github.com/openvenues/libpostal
from utils.settings import get_project_root

ROOT = get_project_root()


def create_address_fields_dataframe(full_address_series=None, unique_street_types=None):

    df_address_fields = pd.DataFrame(columns=['street_number', 'street_type', 'street_name', 'address_line2',
                                              'postal_code', 'city'])

    for idx, add in enumerate(full_address_series):
        if idx % 10000 == 0:
            print('{} entries completed...'.format(idx))
        str_num = str_type = str_name = addline2 = postalcode = city = np.nan

        if type(add) == float:
            pass
        else:
            addline2 = ''
            parsed_add = parse_address(add)

            for item in parsed_add:
                if item[1] == 'city':
                    city = item[0]
                elif item[1] == 'postcode':
                    postalcode = item[0]
                elif item[1] == 'road':
                    road_comps = item[0].split()
                    str_name = ' '.join(road_comps[1:])
                    if road_comps[0] == 'r':
                        str_type = 'rue'
                    elif road_comps[0] in unique_street_types:
                        str_type = road_comps[0]
                    else:
                        str_name = item[0]
                elif item[1] == 'house_number':
                    try:
                        str_num = int(item[0])
                    except ValueError:
                        # print('The parsed \'house_number\' is not a valid integer. Extracting the first integer '\
                        #     'occurrence (from string) and setting it as \'street_number\'')
                        ints = re.findall(r'\d+', item[0])
                        if len(ints) == 0:
                            str_num = np.nan
                        else:
                            sep_ints = [int(s) for s in item[0].split() if s.isdigit()]
                            if len(sep_ints) != 0:
                                str_num = sep_ints[0]
                            else:
                                str_num = ints[0]
                else:
                    addline2 += item[0]
                    addline2 += ' '
            if addline2 == '':
                addline2 = np.nan

        df_address_fields.loc[idx] = [str_num, str_type, str_name, addline2, postalcode, city]

    return df_address_fields


if __name__ == '__main__':
    s1_cstr = pd.read_csv(os.path.join(ROOT, 'data' 'source1_cstr.csv'))
    s2_nond_cstr = pd.read_csv(os.path.join(ROOT, 'data', 'source2_nonanids_nod_cstr.csv'))
    unique_street_types = np.unique(s1_cstr['street_type'].dropna())

    df_address_fields = create_address_fields_dataframe(s2_nond_cstr['address'], unique_street_types)
    df_address_fields.to_csv(os.path.join(ROOT, 'data', 'source2_nonanids_nod_cstr_parsed.csv'), index=False)


