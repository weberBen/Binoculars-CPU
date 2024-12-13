import os

def cast_float(key, default=""):
  item =  os.getenv(key, default)
  item = float(item)
  assert(type(item) is float)

  return item

def cast_int(key, default=""):
  item =  os.getenv(key, default)
  item = int(item)
  assert(type(item) is int)

  return item

def cast_string(key, default="", require=False, strip=True):
  item = os.getenv(key, default)
  if strip:
     item.strip()
  
  if require:
    assert(len(item) > 0)
  
  return item

def cast_list(key, default="", separator="|", strip=True):
    base_item = os.getenv(key, default).split(separator)
    item = []
    for x in base_item:
      if strip:
        x = x.strip()
      
      if len(x) == 0:
        continue

      item.append(x)
        
    return item

def cast_bool(key, default="false"):
  return os.getenv(key, default).lower() in ("true", "1", "yes")