from collections import OrderedDict

import attr
from typing import List  # noqa


@attr.s
class RTCRtpCodecParameters:
    """
    The :class:`RTCRtpCodecParameters` dictionary provides information on
    codec settings.
    """
    name = attr.ib(type=str)  # type: str
    "The codec MIME subtype, for instance `'PCMU'`."
    clockRate = attr.ib(type=int)  # type: int
    "The codec clock rate expressed in Hertz."
    channels = attr.ib(default=None)  # type: int
    "The number of channels supported (e.g. two for stereo)."
    payloadType = attr.ib(default=None)  # type: int
    "The value that goes in the RTP Payload Type Field."
    rtcpFeedback = attr.ib(default=attr.Factory(list))  # type: List[RTCRtcpFeedback]
    "Transport layer and codec-specific feedback messages for this codec."
    parameters = attr.ib(default=attr.Factory(OrderedDict))  # type: OrderedDict
    "Codec-specific parameters available for signaling."

    def __str__(self):
        s = '%s/%d' % (self.name, self.clockRate)
        if self.channels == 2:
            s += '/2'
        return s


@attr.s
class RTCRtpHeaderExtensionParameters:
    """
    The :class:`RTCRtpHeaderExtensionParameters` dictionary enables a header
    extension to be configured for use within an :class:`RTCRtpSender` or
    :class:`RTCRtpReceiver`.
    """
    id = attr.ib(type=int)  # type: int
    "The value that goes in the packet."
    uri = attr.ib(type=str)  # type: str
    "The URI of the RTP header extension."


@attr.s
class RTCRtpCapabilities:
    codecs = attr.ib(default=attr.Factory(list))


@attr.s
class RTCRtcpFeedback:
    """
    The :class:`RTCRtcpFeedback` dictionary provides information on RTCP feedback messages.
    """
    type = attr.ib()  # type: str
    parameter = attr.ib(default=None)  # type: str


@attr.s
class RTCRtcpParameters:
    """
    The :class:`RTCRtcpParameters` dictionary provides information on RTCP settings.
    """
    cname = attr.ib(default=None)  # type: str
    "The Canonical Name (CNAME) used by RTCP."
    mux = attr.ib(default=False)  # type: bool
    "Whether RTP and RTCP are multiplexed."
    ssrc = attr.ib(default=None)  # type: int
    "The Synchronization Source identifier."


@attr.s
class RTCRtpParameters:
    """
    The :class:`RTCRtpParameters` dictionary describes the configuration of
    an :class:`RTCRtpReceiver` or an :class:`RTCRtpSender`.
    """
    codecs = attr.ib(default=attr.Factory(list))  # type: List[RTCRtpCodecParameters]
    "A list of :class:`RTCRtpCodecParameters` to send or receive."
    headerExtensions = attr.ib(
        default=attr.Factory(list))  # type: List[RTCRtpHeaderExtensionParameters]
    "A list of :class:`RTCRtpHeaderExtensionParameters`."
    muxId = attr.ib(default='')  # type: str
    "The muxId assigned to the RTP stream, if any, empty string if unset."
    rtcp = attr.ib(default=attr.Factory(RTCRtcpParameters))  # type: RTCRtcpParameters
    "Parameters to configure RTCP."
