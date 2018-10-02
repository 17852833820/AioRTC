import logging

logger = logging.getLogger('rtp')


class RTCRtpTransceiver:
    """
    The RTCRtpTransceiver interface describes a permanent pairing of an
    :class:`RTCRtpSender` and an :class:`RTCRtpReceiver`, along with some
    shared state.
    """

    def __init__(self, kind, receiver, sender, direction='sendrecv'):
        self.mid = None
        self.__direction = direction
        self.__kind = kind
        self.__receiver = receiver
        self.__sender = sender

    @property
    def direction(self):
        return self.__direction

    @property
    def kind(self):
        return self.__kind

    @property
    def receiver(self):
        """
        The :class:`RTCRtpReceiver` that handles receiving and decoding
        incoming media.
        """
        return self.__receiver

    @property
    def sender(self):
        """
        The :class:`RTCRtpSender` responsible for encoding and sending
        data to the remote peer.
        """
        return self.__sender

    async def stop(self):
        """
        Permanently stops the :class:`RTCRtpTransceiver`.
        """
        await self.__receiver.stop()
        await self.__sender.stop()

    def _set_direction(self, value):
        self.__direction = value
