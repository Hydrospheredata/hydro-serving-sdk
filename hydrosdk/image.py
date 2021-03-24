import re

from collections import namedtuple
from typing import List, Optional


DockerImagePrototype = namedtuple('DockerImage', ['full_name', 'name', 'domain', 'tag', 'digest'])


def _literal(s: str) -> re.Pattern: 
    # _literal compiles s into a literal regular expression, escaping any regexp
    # reserved characters
    return re.compile(re.escape(s))


def _expression(*res: List[re.Pattern]) -> re.Pattern:
    # _expression defines a full expression, where each regular expression must
    # follow the previous.
    return re.compile("".join([r.pattern for r in res]))


def _repeated(*res: List[re.Pattern]) -> re.Pattern:
    # _repeated wraps the regexp in a non-capturing group to get one or more
    # matches.
    return re.compile("(?:" + _expression(*res).pattern + ")+")


def _optional(*res: List[re.Pattern]) -> re.Pattern:
    # _optional wraps the expression in a non-capturing group and makes the
    # production optional.
    return re.compile("(?:" + _expression(*res).pattern + ")?")


def _capture(name, *res: List[re.Pattern]) -> re.Pattern:
    # _capture wraps the expression in a capturing group.
    return re.compile(f"(?P<{name}>" + _expression(*res).pattern + ")")


class DockerImage(DockerImagePrototype):
    # alpha_numeric_regex defines the alpha numeric atom, typically a
    # component of names. This only allows lower case characters and digits.
    alpha_numeric_regex = re.compile("[a-z0-9]+")

    # separator_regex defines the separators allowed to be embedded in name
    # components. This allow one period, one or two underscore and multiple
    # dashes.
    separator_regex = re.compile("(?:[._]|__|[-]*)")

    # name_component_regex restricts registry path component names to start
    # with at least one letter or number, with following parts able to be
    # separated by one period, one or two underscore and multiple dashes.
    name_component_regex = _expression(
        alpha_numeric_regex,
        _optional(
            _repeated(separator_regex, alpha_numeric_regex)
        )
    )

    # domain_component_regex restricts the registry domain component of a
    # repository name to start with a component as defined by DomainRegexp
    # and followed by an optional port.
    domain_component_regex = re.compile("(?:[a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9]|[a-zA-Z0-9])")

    # domain_regex defines the structure of potential domain components
    # that may be part of image names. This is purposely a subset of what is
    # allowed by DNS to ensure backwards compatibility with Docker image
    # names.
    domain_regex = _expression(
        domain_component_regex,
        _optional(_repeated(_literal("."), domain_component_regex)),
        _optional(_literal(":"), re.compile("[0-9]+"))
    )
    
    # tag_regex matches valid tag names. From docker/docker:graph/tags.go.
    tag_regex = re.compile("[\\w][\\w.-]{0,127}")

    # digest_regex matches valid digests.
    digest_regex = re.compile("[A-Za-z][A-Za-z0-9]*(?:[-_+.][A-Za-z][A-Za-z0-9]*)*[:][[:xdigit:]]{32,}")

    # name_regex is the format for the name component of references. The
    # regexp has capturing groups for the domain and name part.
    name_regex = _expression(
        _optional(_capture("domain", domain_regex)), 
        _literal("/"),
        _capture(
            "name",
            name_component_regex,
            _optional(_repeated(_literal("/"), name_component_regex))
        )
    )

    # anchored_name_regex is used to parse a name value, capturing the
    # domain and trailing components.
    anchored_name_regex = _expression(
        _optional(_capture("domain", domain_regex), _literal("/")), 
        _capture(
            "name",
            name_component_regex,
            _optional(_repeated(_literal("/"), name_component_regex))
        )
    )
    
    # identifier_regex is the format for string identifier used as a
    # content addressable identifier using sha256. These identifiers
    # are like digests without the algorithm, since sha256 is used.
    identifier_regex = _capture(
        "identifier",
        re.compile("[a-f0-9]{64}"),
    )

    # short_identifier_regex is the format used to represent a prefix
    # of an identifier. A prefix may be used to match a sha256 identifier
    # within a list of trusted identifiers.
    short_identifier_regex = _capture(
        "short_identifier",
        re.compile("[a-f0-9]{6,64}"),
    )

    # reference_regex is the full supported format of a reference. The regexp
    # is anchored and has capturing groups for name, tag, and digest
    # components.
    reference_regex = _expression(
        _capture("full_name", anchored_name_regex),
        _optional(_literal(":"), _capture("tag", tag_regex)),
        _optional(_literal("@"), _capture("digest", digest_regex))
    )
        
    @classmethod
    def from_string(cls, string):
        matched = cls.reference_regex.match(string)
        if matched is None:
            raise ValueError(f"Couldn't create a DockerImage from the provided image reference: {string}")
        return cls(**matched.groupdict())
    
    @classmethod
    def from_custom(cls, in_dict):
        reference = f'{in_dict["name"]}:{in_dict["tag"]}'
        digest = in_dict.get("sha256")
        if digest is not None:
            reference = f'{reference}@sha256:{digest}'
        return cls.from_string(reference)
    
    def to_string(self):
        string = self.full_name
        if self.tag:
            string = f"{string}:{self.tag}"
        if self.digest:
            string = f"{string}@{self.digest}"
        return string