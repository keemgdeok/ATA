import os
from quantrl import settings

# Construct the path using the corrected BASE_DIR
feature_list_path = os.path.join(settings.BASE_DIR, 'conf', 'data', 'v4.1', 'feature_list.txt')

print(feature_list_path)

# This should print /mnt/c/Users/keemg/dev/qd/conf/data/v4.1/feature_list.txt
