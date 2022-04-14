
def create_name_address_lookup(row=None):
    concatenated_row = str(row.name) + ' ' + str(row.street_number) + ' ' + str(row.street_type) + ' ' + \
                       str(row.street_name) + ' ' + str(row.address_line2) + ' ' + str(row.postal_code) + ' ' + str(row.city)
    return concatenated_row
