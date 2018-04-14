import attr


@attr.s
class RTCRtpCodecParameters:
    """
    The :class:`RTCRtpCodecParameters` dictionary provides information on
    codec settings.
    """
    name = attr.ib(type=str)
    "The codec MIME subtype, for instance `'PCMU'`."
    clockRate = attr.ib(type=int)
    "The codec clock rate expressed in Hertz."
    channels = attr.ib(default=None)
    "The number of channels supported (e.g. two for stereo)."
    payloadType = attr.ib(default=None)
    "The value that goes in the RTP Payload Type Field."

    def clone(self, payloadType):
        return RTCRtpCodecParameters(
            name=self.name, clockRate=self.clockRate,
            channels=self.channels, payloadType=payloadType)

    def __str__(self):
        s = '%s/%d' % (self.name, self.clockRate)
        if self.channels == 2:
            s += '/2'
        return s


@attr.s
class RTCRtpCapabilities:
    codecs = attr.ib(default=attr.Factory(list))


@attr.s
class RTCRtcpParameters:
    """
    The :class:`RTCRtcpParameters` dictionary  provides information on RTCP settings.
    """
    cname = attr.ib(default=None)
    "The Canonical Name (CNAME) used by RTCP."
    mux = attr.ib(default=False)
    "Whether RTP and RTCP are multiplexed."
    ssrc = attr.ib(default=None)
    "The Synchronization Source identifier."


@attr.s
class RTCRtpParameters:
    """
    The :class:`RTCRtpParameters` dictionary describes the configuration of
    an :class:`RTCRtpReceiver` or an :class:`RTCRtpSender`.
    """
    codecs = attr.ib(default=attr.Factory(list))
    "A list of :class:`RTCRtpCodecParameters` to send or receive."
    muxId = attr.ib(default='')
    "The muxId assigned to the RTP stream, if any, empty string if unset."
    rtcp = attr.ib(default=attr.Factory(RTCRtcpParameters))
    "Parameters to configure RTCP."
