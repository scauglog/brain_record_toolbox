import yaml

class Settings: 
  def __init__(self, path = "../settings/settings.yaml"):
    with open(path, 'r') as fh:
         self.settings = yaml.load(fh)

  def get(self):
    return self.settings