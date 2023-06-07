from Prepare import Prepare

configs = [
    {
        "anotation_method": "blip",  # choices=["blip", "wd14-tagger", "both"], help="anotation method""--train_data_dir", type=str, default="", help="directory for train images")
        "max_length": 75,  # type=int, default=75, help="max length of caption ")
        "min_length": 5,  # type=int, default=5, help="min length of caption ")
        "general_threshold": 0.3,  # type=float, default=0.3, help="threshold of confidence to add a tag for general category, same as --thresh if omitted ")
        "character_threshold": 0.3,  # type=float, default=0.3, help="threshold of confidence to add a tag for character category, same as --thres if omitted ")
        "undesired_tags": "",  # type=str, default="", help="comma-separated list of undesired tags to remove from the output ")
        "tags_to_replace": "",  # type=str, default="", help="comma-separated list of tags to replace with a new tag ")
        "tags_to_add_to_front": "",  # type=str, default="", help="comma-separated list of tags to add to the front of the output ")
    }
]

for config in configs:
    model = Prepare(**config)
    model.run()
