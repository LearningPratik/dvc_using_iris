```mermaid
flowchart TD
	node1["data_process"]
	node2["data_split"]
	node3["evaluate"]
	node4["train"]
	node1-->node3
	node1-->node4
	node2-->node1
	node4-->node3
	node5["dvc\data\iris.csv.dvc"]
```
