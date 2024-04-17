encoding_mappings = {}

# # Encode each variable and store mappings
# encoded_variables = ['gender', 'stream', 'college_name', 'placement_status']
# for variable in encoded_variables:
#     jobData[f'{variable}'] = le.fit_transform(jobData[variable])
#     encoding_mappings[variable] = dict(zip(le.classes_, le.transform(le.classes_)))