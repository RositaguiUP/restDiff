import json

# Load the JSON
with open('/home/rosita/tests/diff/restDiff/data/2F5Z7_007/1/poses.json', 'r') as f:
    data = json.load(f)

# Update each frame
for frame in data.get('frames', []):
    frame['orientation'] = 'portrait'

# Save the updated JSON
with open('/home/rosita/tests/diff/restDiff/data/2F5Z7_007/1/poses.json', 'w') as f:
    json.dump(data, f, indent=4)