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
    
<<<<<<< HEAD
    def to_string(self):
=======
    def __str__(self):
>>>>>>> 62e41637cb25e4e8dd00671e1b9c42c70eff1cb6
        return f"{self.name}:{self.tag}"
