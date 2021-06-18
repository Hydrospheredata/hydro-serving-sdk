from docker_image.reference import TaggedReference
from pydantic import BaseModel

class DockerImage(BaseModel):
    name: str
    tag: str
    
    @classmethod
    def from_string(cls, string):
        ref = TaggedReference.parse(string)
        if isinstance(ref, TaggedReference):
            return DockerImage(name = ref['name'], tag = ref['tag'])
        else:
            raise ValueError(f"Couldn't create a DockerImage from the provided image reference: {string}")

    def __str__(self):
        return f"{self.name}:{self.tag}"
