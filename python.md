## Iterate over files in a directory (`.xml` in the example)
```python
file_list = [file for file in os.scandir(dir_to_parse) if file.name.split('.')[-1] == 'xml']
```
