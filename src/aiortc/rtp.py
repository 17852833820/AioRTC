import math
import os
import struct
from dataclasses import dataclass, field
from struct import pack, unpack, unpack_from
from typing import Any, List, Optional, Tuple, Union
from enum import Enum,auto

from av import AudioFrame

from .rtcrtpparameters import RTCRtpParameters

# used for NACK and retransmission
RTP_HISTORY_SIZE = 128

# reserved to avoid confusion with RTCP
FORBIDDEN_PAYLOAD_TYPES = range(72, 77)
DYNAMIC_PAYLOAD_TYPES = range(96, 128)

RTP_HEADER_LENGTH = 13 # 原始12字节，增加1字节
RTCP_HEADER_LENGTH = 4

PACKETS_LOST_MIN = -(1 << 23)
PACKETS_LOST_MAX = (1 << 23) - 1

RTCP_SR = 200
RTCP_RR = 201
RTCP_SDES = 202
RTCP_BYE = 203
RTCP_APP = 204
RTCP_RTPFB = 205
RTCP_PSFB = 206

RTCP_RTPFB_NACK = 1

RTCP_PSFB_PLI = 1
RTCP_PSFB_SLI = 2
RTCP_PSFB_RPSI = 3
RTCP_PSFB_IF = 12
RTCP_PSFB_APP = 15


@dataclass
class HeaderExtensions:
    abs_send_time: Optional[int] = None
    audio_level: Any = None
    mid: Any = None
    repaired_rtp_stream_id: Any = None
    rtp_stream_id: Any = None
    transmission_offset: Optional[int] = None
    transport_sequence_number: Optional[int] = None
    marker_first=None


class HeaderExtensionsMap:
    def __init__(self) -> None:
        self.__ids = HeaderExtensions()

    def configure(self, parameters: RTCRtpParameters) -> None:
        for ext in parameters.headerExtensions:
            if ext.uri == "urn:ietf:params:rtp-hdrext:sdes:mid":
                self.__ids.mid = ext.id
            # 新增扩展信息marker_first
            if ext.uri == "urn:ietf:params:rtp-hdrext:sdes:marker_first": 
                self.__ids.marker_first = ext.id
            elif ext.uri == "urn:ietf:params:rtp-hdrext:sdes:repaired-rtp-stream-id":
                self.__ids.repaired_rtp_stream_id = ext.id
            elif ext.uri == "urn:ietf:params:rtp-hdrext:sdes:rtp-stream-id":
                self.__ids.rtp_stream_id = ext.id
            elif (
                ext.uri == "http://www.webrtc.org/experiments/rtp-hdrext/abs-send-time"
            ):
                self.__ids.abs_send_time = ext.id
            elif ext.uri == "urn:ietf:params:rtp-hdrext:toffset":
                self.__ids.transmission_offset = ext.id
            elif ext.uri == "urn:ietf:params:rtp-hdrext:ssrc-audio-level":
                self.__ids.audio_level = ext.id
            elif (
                ext.uri
                == "http://www.ietf.org/id/draft-holmer-rmcat-transport-wide-cc-extensions-01"
            ):
                self.__ids.transport_sequence_number = ext.id

    def get(self, extension_profile: int, extension_value: bytes) -> HeaderExtensions:
        values = HeaderExtensions()
        for x_id, x_value in unpack_header_extensions(
            extension_profile, extension_value
        ):
            if x_id == self.__ids.mid:
                values.mid = x_value.decode("utf8")
            elif x_id == self.__ids.marker_first:
                values.marker_first = x_value.decode("utf8")
            elif x_id == self.__ids.repaired_rtp_stream_id:
                values.repaired_rtp_stream_id = x_value.decode("ascii")
            elif x_id == self.__ids.rtp_stream_id:
                values.rtp_stream_id = x_value.decode("ascii")
            elif x_id == self.__ids.abs_send_time:
                values.abs_send_time = unpack("!L", b"\00" + x_value)[0]
            elif x_id == self.__ids.transmission_offset:
                values.transmission_offset = unpack("!l", x_value + b"\00")[0] >> 8
            elif x_id == self.__ids.audio_level:
                vad_level = unpack("!B", x_value)[0]
                values.audio_level = (vad_level & 0x80 == 0x80, vad_level & 0x7F)
            elif x_id == self.__ids.transport_sequence_number:
                values.transport_sequence_number = unpack("!H", x_value)[0]
        return values

    def set(self, values: HeaderExtensions):
        extensions = []
        if values.mid is not None and self.__ids.mid:
            extensions.append((self.__ids.mid, values.mid.encode("utf8")))
        if (values.marker_first is not None and self.__ids.marker_first):
            extensions.append((self.__ids.marker_first,values.marker_first.encode("utf8")))
        if (
            values.repaired_rtp_stream_id is not None
            and self.__ids.repaired_rtp_stream_id
        ):
            extensions.append(
                (
                    self.__ids.repaired_rtp_stream_id,
                    values.repaired_rtp_stream_id.encode("ascii"),
                )
            )
        if values.rtp_stream_id is not None and self.__ids.rtp_stream_id:
            extensions.append(
                (self.__ids.rtp_stream_id, values.rtp_stream_id.encode("ascii"))
            )
        if values.abs_send_time is not None and self.__ids.abs_send_time:
            extensions.append(
                (self.__ids.abs_send_time, pack("!L", values.abs_send_time)[1:])
            )
        if values.transmission_offset is not None and self.__ids.transmission_offset:
            extensions.append(
                (
                    self.__ids.transmission_offset,
                    pack("!l", values.transmission_offset << 8)[0:2],
                )
            )
        if values.audio_level is not None and self.__ids.audio_level:
            extensions.append(
                (
                    self.__ids.audio_level,
                    pack(
                        "!B",
                        (0x80 if values.audio_level[0] else 0)
                        | (values.audio_level[1] & 0x7F),
                    ),
                )
            )
        if (
            values.transport_sequence_number is not None
            and self.__ids.transport_sequence_number
        ):
            extensions.append(
                (
                    self.__ids.transport_sequence_number,
                    pack("!H", values.transport_sequence_number),
                )
            )
        return pack_header_extensions(extensions)


def clamp_packets_lost(count: int) -> int:
    return max(PACKETS_LOST_MIN, min(count, PACKETS_LOST_MAX))


def pack_packets_lost(count: int) -> bytes:
    return pack("!l", count)[1:]


def unpack_packets_lost(d: bytes) -> int:
    if d[0] & 0x80:
        d = b"\xff" + d
    else:
        d = b"\x00" + d
    return unpack("!l", d)[0]


def pack_rtcp_packet(packet_type: int, count: int, payload: bytes) -> bytes:
    assert len(payload) % 4 == 0
    return pack("!BBH", (2 << 6) | count, packet_type, len(payload) // 4) + payload
#扩展反馈包
def pack_if_fci(isFinish: int, ssrcs: List[int]) -> bytes:
    """
    Pack the FCI for a Receiver Estimated Maximum Bitrate report.

    https://tools.ietf.org/html/draft-alvestrand-rmcat-remb-03
    """
    data = b"IF"
    
    isFinish &= 0x01
    data +=pack("!B", isFinish)
    
    
    return data

def pack_remb_fci(bitrate: int, ssrcs: List[int]) -> bytes:
    """
    Pack the FCI for a Receiver Estimated Maximum Bitrate report.

    https://tools.ietf.org/html/draft-alvestrand-rmcat-remb-03
    """
    data = b"REMB"
    exponent = 0
    mantissa = bitrate
    while mantissa > 0x3FFFF:
        mantissa >>= 1#表示一个尾数
        exponent += 1#表示一个指数
    data += pack( 
        "!BBH", len(ssrcs), (exponent << 2) | (mantissa >> 16), (mantissa & 0xFFFF)
    )
    for ssrc in ssrcs:
        data += pack("!L", ssrc)
    return data

def unpack_if_fci(data: bytes) -> int:
    """
    Unpack the FCI for a Receiver Estimated Maximum Bitrate report.

    https://tools.ietf.org/html/draft-alvestrand-rmcat-remb-03
    """
    if len(data) < 3 or data[0:4] != b"IF":
        raise ValueError("Invalid IF prefix")
    isFinish=unpack("!B", data)[3] & 0x01

    return (isFinish)
def unpack_remb_fci(data: bytes) -> Tuple[int, List[int]]:
    """
    Unpack the FCI for a Receiver Estimated Maximum Bitrate report.

    https://tools.ietf.org/html/draft-alvestrand-rmcat-remb-03
    """
    if len(data) < 8 or data[0:4] != b"REMB":
        raise ValueError("Invalid REMB prefix")

    exponent = (data[5] & 0xFC) >> 2 #exponent 从 data[5] 的第 2 到第 7 位得出
    mantissa = ((data[5] & 0x03) << 16) | (data[6] << 8) | data[7] #mantissa 使用 data[5] 的第 0 到第 1 位，data[6] 和 data[7] 构建。它通过将 data[5] 的位左移 16 位，将 data[6] 的位左移 8 位，以及直接使用 data[7] 的位，然后使用 | 运算符组合这些位。
    bitrate = mantissa << exponent #将 mantissa 左移 exponent 位得到

    pos = 8
    ssrcs = []
    for r in range(data[4]):
        ssrcs.append(unpack_from("!L", data, pos)[0])
        pos += 4

    return (bitrate, ssrcs)


def is_rtcp(msg: bytes) -> bool:
    return len(msg) >= 2 and msg[1] >= 192 and msg[1] <= 208


def padl(length: int) -> int:
    """
    Return amount of padding needed for a 4-byte multiple.
    """
    return 4 * ((length + 3) // 4) - length


def unpack_header_extensions(
    extension_profile: int, extension_value: bytes
) -> List[Tuple[int, bytes]]:
    """
    Parse header extensions according to RFC 5285.
    """
    extensions = []
    pos = 0

    if extension_profile == 0xBEDE:
        # One-Byte Header
        while pos < len(extension_value):
            # skip padding byte
            if extension_value[pos] == 0:
                pos += 1
                continue

            x_id = (extension_value[pos] & 0xF0) >> 4
            x_length = (extension_value[pos] & 0x0F) + 1
            pos += 1

            if len(extension_value) < pos + x_length:
                raise ValueError("RTP one-byte header extension value is truncated")
            x_value = extension_value[pos : pos + x_length]
            extensions.append((x_id, x_value))
            pos += x_length
    elif extension_profile == 0x1000:
        # Two-Byte Header
        while pos < len(extension_value):
            # skip padding byte
            if extension_value[pos] == 0:
                pos += 1
                continue

            if len(extension_value) < pos + 2:
                raise ValueError("RTP two-byte header extension is truncated")
            x_id, x_length = unpack_from("!BB", extension_value, pos)
            pos += 2

            if len(extension_value) < pos + x_length:
                raise ValueError("RTP two-byte header extension value is truncated")
            x_value = extension_value[pos : pos + x_length]
            extensions.append((x_id, x_value))
            pos += x_length

    return extensions


def pack_header_extensions(extensions: List[Tuple[int, bytes]]) -> Tuple[int, bytes]:
    """
    Serialize header extensions according to RFC 5285.
    """
    extension_profile = 0
    extension_value = b""

    if not extensions:
        return extension_profile, extension_value

    one_byte = True
    for x_id, x_value in extensions:
        x_length = len(x_value)
        assert x_id > 0 and x_id < 256
        assert x_length >= 0 and x_length < 256
        if x_id > 14 or x_length == 0 or x_length > 16:
            one_byte = False

    if one_byte:
        # One-Byte Header
        extension_profile = 0xBEDE
        extension_value = b""
        for x_id, x_value in extensions:
            x_length = len(x_value)
            extension_value += pack("!B", (x_id << 4) | (x_length - 1))
            extension_value += x_value
    else:
        # Two-Byte Header
        extension_profile = 0x1000
        extension_value = b""
        for x_id, x_value in extensions:
            x_length = len(x_value)
            extension_value += pack("!BB", x_id, x_length)
            extension_value += x_value

    extension_value += b"\x00" * padl(len(extension_value))
    return extension_profile, extension_value


def compute_audio_level_dbov(frame: AudioFrame):
    """
    Compute the energy level as spelled out in RFC 6465, Appendix A.
    """
    MAX_SAMPLE_VALUE = 32767
    MAX_AUDIO_LEVEL = 0
    MIN_AUDIO_LEVEL = -127
    rms = 0
    buf = bytes(frame.planes[0])
    s = struct.Struct("h")
    for unpacked in s.iter_unpack(buf):
        sample = unpacked[0]
        rms += sample * sample
    rms = math.sqrt(rms / (frame.samples * MAX_SAMPLE_VALUE * MAX_SAMPLE_VALUE))
    if rms > 0:
        db = 20 * math.log10(rms)
        db = max(db, MIN_AUDIO_LEVEL)
        db = min(db, MAX_AUDIO_LEVEL)
    else:
        db = MIN_AUDIO_LEVEL
    return round(db)


@dataclass
class RtcpReceiverInfo:
    ssrc: int
    fraction_lost: int
    packets_lost: int
    highest_sequence: int
    jitter: int
    lsr: int
    dlsr: int

    def __bytes__(self) -> bytes:
        data = pack("!LB", self.ssrc, self.fraction_lost)
        data += pack_packets_lost(self.packets_lost)
        data += pack("!LLLL", self.highest_sequence, self.jitter, self.lsr, self.dlsr)
        return data

    @classmethod
    def parse(cls, data: bytes):
        ssrc, fraction_lost = unpack("!LB", data[0:5])
        packets_lost = unpack_packets_lost(data[5:8])
        highest_sequence, jitter, lsr, dlsr = unpack("!LLLL", data[8:])
        return cls(
            ssrc=ssrc,
            fraction_lost=fraction_lost,
            packets_lost=packets_lost,
            highest_sequence=highest_sequence,
            jitter=jitter,
            lsr=lsr,
            dlsr=dlsr,
        )


@dataclass
class RtcpSenderInfo:
    ntp_timestamp: int
    rtp_timestamp: int
    packet_count: int
    octet_count: int

    def __bytes__(self) -> bytes:
        return pack(
            "!QLLL",
            self.ntp_timestamp,
            self.rtp_timestamp,
            self.packet_count,
            self.octet_count,
        )

    @classmethod
    def parse(cls, data: bytes):
        ntp_timestamp, rtp_timestamp, packet_count, octet_count = unpack("!QLLL", data)
        return cls(
            ntp_timestamp=ntp_timestamp,
            rtp_timestamp=rtp_timestamp,
            packet_count=packet_count,
            octet_count=octet_count,
        )


@dataclass
class RtcpSourceInfo:
    ssrc: int
    items: List[Tuple[Any, bytes]]


@dataclass
class RtcpByePacket:
    sources: List[int]

    def __bytes__(self) -> bytes:
        payload = b"".join([pack("!L", ssrc) for ssrc in self.sources])
        return pack_rtcp_packet(RTCP_BYE, len(self.sources), payload)

    @classmethod
    def parse(cls, data: bytes, count: int):
        if len(data) < 4 * count:
            raise ValueError("RTCP bye length is invalid")
        if count > 0:
            sources = list(unpack_from("!" + ("L" * count), data, 0))
        else:
            sources = []
        return cls(sources=sources)


@dataclass
class RtcpPsfbPacket:
    """
    Payload-Specific Feedback Message (RFC 4585).
    """

    fmt: int
    ssrc: int
    media_ssrc: int
    fci: bytes = b""

    def __bytes__(self) -> bytes:
        payload = pack("!LL", self.ssrc, self.media_ssrc) + self.fci
        return pack_rtcp_packet(RTCP_PSFB, self.fmt, payload)

    @classmethod
    def parse(cls, data: bytes, fmt: int):
        if len(data) < 8:
            raise ValueError("RTCP payload-specific feedback length is invalid")

        ssrc, media_ssrc = unpack("!LL", data[0:8])
        fci = data[8:]
        return cls(fmt=fmt, ssrc=ssrc, media_ssrc=media_ssrc, fci=fci)


@dataclass
class RtcpRrPacket:
    ssrc: int
    reports: List[RtcpReceiverInfo] = field(default_factory=list)

    def __bytes__(self) -> bytes:
        payload = pack("!L", self.ssrc)
        for report in self.reports:
            payload += bytes(report)
        return pack_rtcp_packet(RTCP_RR, len(self.reports), payload)

    @classmethod
    def parse(cls, data: bytes, count: int):
        if len(data) != 4 + 24 * count:
            raise ValueError("RTCP receiver report length is invalid")

        ssrc = unpack("!L", data[0:4])[0]
        pos = 4
        reports = []
        for r in range(count):
            reports.append(RtcpReceiverInfo.parse(data[pos : pos + 24]))
            pos += 24
        return cls(ssrc=ssrc, reports=reports)


@dataclass
class RtcpRtpfbPacket:
    """
    Generic RTP Feedback Message (RFC 4585).
    """

    fmt: int
    ssrc: int
    media_ssrc: int

    # generick NACK
    lost: List[int] = field(default_factory=list)

    def __bytes__(self) -> bytes:
        payload = pack("!LL", self.ssrc, self.media_ssrc)
        if self.lost:
            pid = self.lost[0]
            blp = 0
            for p in self.lost[1:]:
                d = p - pid - 1
                if d < 16:
                    blp |= 1 << d
                else:
                    payload += pack("!HH", pid, blp)
                    pid = p
                    blp = 0
            payload += pack("!HH", pid, blp)
        return pack_rtcp_packet(RTCP_RTPFB, self.fmt, payload)

    @classmethod
    def parse(cls, data: bytes, fmt: int):
        if len(data) < 8 or len(data) % 4:
            raise ValueError("RTCP RTP feedback length is invalid")

        ssrc, media_ssrc = unpack("!LL", data[0:8])
        lost = []
        for pos in range(8, len(data), 4):
            pid, blp = unpack("!HH", data[pos : pos + 4])
            lost.append(pid)
            for d in range(0, 16):
                if (blp >> d) & 1:
                    lost.append(pid + d + 1)
        return cls(fmt=fmt, ssrc=ssrc, media_ssrc=media_ssrc, lost=lost)


@dataclass
class RtcpSdesPacket:
    chunks: List[RtcpSourceInfo] = field(default_factory=list)

    def __bytes__(self) -> bytes:
        payload = b""
        for chunk in self.chunks:
            payload += pack("!L", chunk.ssrc)
            for d_type, d_value in chunk.items:
                payload += pack("!BB", d_type, len(d_value)) + d_value
            payload += b"\x00\x00"
        while len(payload) % 4:
            payload += b"\x00"
        return pack_rtcp_packet(RTCP_SDES, len(self.chunks), payload)

    @classmethod
    def parse(cls, data: bytes, count: int):
        pos = 0
        chunks = []
        for r in range(count):
            if len(data) < pos + 4:
                raise ValueError("RTCP SDES source is truncated")
            ssrc = unpack_from("!L", data, pos)[0]
            pos += 4

            items = []
            while pos < len(data) - 1:
                d_type, d_length = unpack_from("!BB", data, pos)
                pos += 2

                if len(data) < pos + d_length:
                    raise ValueError("RTCP SDES item is truncated")
                d_value = data[pos : pos + d_length]
                pos += d_length
                if d_type == 0:
                    break
                else:
                    items.append((d_type, d_value))
            chunks.append(RtcpSourceInfo(ssrc=ssrc, items=items))
        return cls(chunks=chunks)


@dataclass
class RtcpSrPacket:
    ssrc: int
    sender_info: RtcpSenderInfo
    reports: List[RtcpReceiverInfo] = field(default_factory=list)

    def __bytes__(self) -> bytes:
        payload = pack("!L", self.ssrc)
        payload += bytes(self.sender_info)
        for report in self.reports:
            payload += bytes(report)
        return pack_rtcp_packet(RTCP_SR, len(self.reports), payload)

    @classmethod
    def parse(cls, data: bytes, count: int):
        if len(data) != 24 + 24 * count:
            raise ValueError("RTCP sender report length is invalid")

        ssrc = unpack_from("!L", data)[0]
        sender_info = RtcpSenderInfo.parse(data[4:24])
        pos = 24
        reports = []
        for r in range(count):
            reports.append(RtcpReceiverInfo.parse(data[pos : pos + 24]))
            pos += 24
        return RtcpSrPacket(ssrc=ssrc, sender_info=sender_info, reports=reports)


AnyRtcpPacket = Union[
    RtcpByePacket,
    RtcpPsfbPacket,
    RtcpRrPacket,
    RtcpRtpfbPacket,
    RtcpSdesPacket,
    RtcpSrPacket,
]


class RtcpPacket:
    @classmethod
    def parse(cls, data: bytes) -> List[AnyRtcpPacket]:
        pos = 0
        packets = []

        while pos < len(data):
            if len(data) < pos + RTCP_HEADER_LENGTH:
                raise ValueError(
                    f"RTCP packet length is less than {RTCP_HEADER_LENGTH} bytes"
                )

            v_p_count, packet_type, length = unpack("!BBH", data[pos : pos + 4])
            version = v_p_count >> 6
            padding = (v_p_count >> 5) & 1
            count = v_p_count & 0x1F
            if version != 2:
                raise ValueError("RTCP packet has invalid version")
            pos += 4

            end = pos + length * 4
            if len(data) < end:
                raise ValueError("RTCP packet is truncated")
            payload = data[pos:end]
            pos = end

            if padding:
                if not payload or not payload[-1] or payload[-1] > len(payload):
                    raise ValueError("RTCP packet padding length is invalid")
                payload = payload[0 : -payload[-1]]

            if packet_type == RTCP_BYE:
                packets.append(RtcpByePacket.parse(payload, count))
            elif packet_type == RTCP_SDES:
                packets.append(RtcpSdesPacket.parse(payload, count))
            elif packet_type == RTCP_SR:
                packets.append(RtcpSrPacket.parse(payload, count))
            elif packet_type == RTCP_RR:
                packets.append(RtcpRrPacket.parse(payload, count))
            elif packet_type == RTCP_RTPFB:
                packets.append(RtcpRtpfbPacket.parse(payload, count))
            elif packet_type == RTCP_PSFB:
                packets.append(RtcpPsfbPacket.parse(payload, count))

        return packets


class RtpPacket:
    def __init__(
        self,
        payload_type: int = 0,
        marker: int = 0,
        sequence_number: int = 0,
        timestamp: int = 0,
        ssrc: int = 0,
        payload: bytes = b"",
    ) -> None:
        self.version = 2
        self.marker = marker
        self.payload_type = payload_type
        self.sequence_number = sequence_number
        self.timestamp = timestamp
        self.ssrc = ssrc
        self.csrc: List[int] = []
        self.extensions = HeaderExtensions()
        self.payload = payload
        self.padding_size = 0
        self.payload_size = 0
        self.header_size=0

    def __repr__(self) -> str:
        return (
            f"RtpPacket(seq={self.sequence_number}, ts={self.timestamp}, "
            f"marker={self.marker}, payload={self.payload_type}, "
            f"{len(self.payload)} bytes)"
        )
    
    # @classmethod
    # def parse(cls, data: bytes, extensions_map=HeaderExtensionsMap()):
    #     if len(data) < RTP_HEADER_LENGTH:
    #         raise ValueError(
    #             f"RTP packet length is less than {RTP_HEADER_LENGTH} bytes"
    #         )

    #     v_p_x_cc, m_pt, sequence_number, timestamp, ssrc = unpack("!BBHLL", data[0:12])
    #     version = v_p_x_cc >> 6
    #     padding = (v_p_x_cc >> 5) & 1
    #     extension = (v_p_x_cc >> 4) & 1
    #     cc = v_p_x_cc & 0x0F
    #     if version != 2:
    #         raise ValueError("RTP packet has invalid version")
    #     if len(data) < RTP_HEADER_LENGTH + 4 * cc:
    #         raise ValueError("RTP packet has truncated CSRC")

    #     packet = cls(
    #         marker=(m_pt >> 7),
    #         payload_type=(m_pt & 0x7F),
    #         sequence_number=sequence_number,
    #         timestamp=timestamp,
    #         ssrc=ssrc,
    #     )

    #     pos = RTP_HEADER_LENGTH
    #     for i in range(0, cc):
    #         packet.csrc.append(unpack_from("!L", data, pos)[0])
    #         pos += 4

    #     if extension:
    #         if len(data) < pos + 4:
    #             raise ValueError("RTP packet has truncated extension profile / length")
    #         extension_profile, extension_length = unpack_from("!HH", data, pos)
    #         extension_length *= 4
    #         pos += 4

    #         if len(data) < pos + extension_length:
    #             raise ValueError("RTP packet has truncated extension value")
    #         extension_value = data[pos : pos + extension_length]
    #         pos += extension_length
    #         packet.extensions = extensions_map.get(extension_profile, extension_value)

    #     if padding:
    #         padding_len = data[-1]
    #         if not padding_len or padding_len > len(data) - pos:
    #             raise ValueError("RTP packet padding length is invalid")
    #         packet.padding_size = padding_len
    #         packet.payload = data[pos:-padding_len]
    #     else:
    #         packet.payload = data[pos:]
    #     packet.payload_size=len(packet.payload)

    #     return packet

    # def serialize(self, extensions_map=HeaderExtensionsMap()) -> bytes:
    #     extension_profile, extension_value = extensions_map.set(self.extensions)
    #     has_extension = bool(extension_value)
    #     self.header_size=len(extension_value)
    #     padding = self.padding_size > 0
    #     data = pack(
    #         "!BBHLL",
    #         (self.version << 6)
    #         | (padding << 5)
    #         | (has_extension << 4)
    #         | len(self.csrc),
    #         (self.marker << 7) | self.payload_type,
    #         self.sequence_number,
    #         self.timestamp,
    #         self.ssrc,
    #     )
    #     for csrc in self.csrc:
    #         data += pack("!L", csrc)
    #     if has_extension:
    #         data += pack("!HH", extension_profile, len(extension_value) >> 2)
    #         data += extension_value
    #     data += self.payload
    #     if padding:
    #         data += os.urandom(self.padding_size - 1)
    #         data += bytes([self.padding_size])
    #     return data





class RtpPacketMediaType(Enum):
    kAudio = auto()                    # Audio media packets.
    kVideo = auto()                    # Video media packets.
    kRetransmission = auto()           # Retransmissions, sent as response to NACK.
    kForwardErrorCorrection = auto()   # FEC packets.
    kPadding = auto()                  # RTX or plain padding sent to maintain BWE.
#############################################################################################################
#+-+-+—+-+-+—+-+-+—+-+-+—+-+-+—+-+-+—+-+-+—+-+-+—+-+-+—+-+-+—+-+-+—+-+-+—+-+-+—+-+-+—+-+-+—+-+-+—+-+-+-+-+-+-
#｜Version(2bit)｜P(1bit)|X(1bit)|  CC(4bit)  | marker(1bit) | payload type(7bit) |  sequence number(16bit)  |                                              
#|is key frame(1bit)｜isFirstPkt(1bit) |                        packet type(6bit)                            |                                              
#｜                                        timestamp(32bit)                                                  |
#｜                                             ssrc(32bit)                                                  |
#｜                                             csrc(32bit)                                                  |
#|    extension_profile(16bit)     |                   len(extension_value)/4(16bit)                         |
#|+-+-+—+-+-+—+-+-+—+           extension_value           +-+-+—+-+-+—+-+-+—+-+-+—+-+-+—+-+-+—+-+-+—+-+-+—+-+|   
#|+-+-+—+-+-+—+-+-+—+           payload                   +-+-+—+-+-+—+-+-+—+-+-+—+-+-+—+-+-+—+-+-+—+-+-+—+-+|   
#|+-+-+—+-+-+—+-+-+—+           padding                   +-+-+—+-+-+—+-+-+—+-+-+—+-+-+—+-+-+—+-+-+—+-+-+—+-+|   
#|+-+-+—+-+-+—+-+-+—+           padding size              +-+-+—+-+-+—+-+-+—+-+-+—+-+-+—+-+-+—+-+-+—+-+-+—+-+|   

 
class RtpPacketToSend(RtpPacket):
    def __init__(self, payload_type: int = 0, marker: int = 0, sequence_number: int = 0, timestamp: int = 0, ssrc: int = 0, payload: bytes = b"") -> None:
        super().__init__(payload_type, marker, sequence_number, timestamp, ssrc, payload)
        self._capture_time_ms:int =0
        self._packet_type:Optional[RtpPacketMediaType]=None
        self._retransmitted_sequence_number: Optional[int] = None
        self._allow_retransmission: bool = False
        self._application_data: List[int] = []
        self._is_first_packet_of_frame: bool = False
        self._is_key_frame: bool = False
    def set_capture_time_ms(self, time: int):
        self._capture_time_ms = time
    def capture_time_ms(self)->int:
        return self._capture_time_ms
    def set_packet_type(self, packet_type: RtpPacketMediaType):
        self._packet_type = packet_type
    def packet_type(self)->RtpPacketMediaType:
        return self._packet_type
    def set_retransmitted_sequence_number(self, sequence_number: int):
        self._retransmitted_sequence_number = sequence_number

    def retransmitted_sequence_number(self) -> Optional[int]:
        return self._retransmitted_sequence_number

    def set_allow_retransmission(self, allow_retransmission: bool):
        self._allow_retransmission = allow_retransmission

    def allow_retransmission(self) -> bool:
        return self._allow_retransmission

    def application_data(self) -> bytes:
        return bytes(self._application_data)
    # 应用程序特定数据
    def set_application_data(self, data: bytes):
        self._application_data = bytearray(data)

    # def set_packetization_finish_time_ms(self, time: int):
    #     delta = VideoSendTiming.GetDeltaCappedMs(self._capture_time_ms, time)
    #     self.set_extension(VideoTimingExtension, delta, VideoTimingExtension.kPacketizationFinishDeltaOffset)

    # def set_pacer_exit_time_ms(self, time: int):
    #     delta = VideoSendTiming.GetDeltaCappedMs(self.capture_time_ms, time)
    #     self.set_extension(VideoTimingExtension, delta, VideoTimingExtension.kPacerExitDeltaOffset)

    # def set_network_time_ms(self, time: int):
    #     delta = VideoSendTiming.GetDeltaCappedMs(self.capture_time_ms, time)
    #     self.set_extension(VideoTimingExtension, delta, VideoTimingExtension.kNetworkTimestampDeltaOffset)

    # def set_network2_time_ms(self, time: int):
    #     delta = VideoSendTiming.GetDeltaCappedMs(self.capture_time_ms, time)
    #     self.set_extension(VideoTimingExtension, delta, VideoTimingExtension.kNetwork2TimestampDeltaOffset)

    def set_first_packet_of_frame(self, is_first_packet: bool):
        self._is_first_packet_of_frame = is_first_packet

    def is_first_packet_of_frame(self) -> bool:
        return self._is_first_packet_of_frame

    def set_is_key_frame(self, is_key_frame: bool):
        self._is_key_frame = is_key_frame

    def is_key_frame(self) -> bool:
        return self._is_key_frame
    def serialize(self, extensions_map=HeaderExtensionsMap()) -> bytes:
        extension_profile, extension_value = extensions_map.set(self.extensions)
        has_extension = bool(extension_value)
        self.header_size=len(extension_value)
        padding = self.padding_size > 0
        data = pack(
            "!BBHBLL",
            (self.version << 6)
            | (padding << 5)
            | (has_extension << 4)
            | len(self.csrc),
            (self.marker << 7) | self.payload_type,
            self.sequence_number,
            (self._is_key_frame<<7)|(self._is_first_packet_of_frame<<6)|self._packet_type.value,
            self.timestamp,
            self.ssrc,
        )
        
        for csrc in self.csrc:
            data += pack("!L", csrc)
        if has_extension:
            data += pack("!HH", extension_profile, len(extension_value) >> 2)
            data += extension_value
        data += self.payload
        if padding:
            data += os.urandom(self.padding_size - 1)
            data += bytes([self.padding_size])
        return data
    @classmethod
    def parse(cls, data: bytes, extensions_map=HeaderExtensionsMap()):
        if len(data) < RTP_HEADER_LENGTH:
            raise ValueError(
                f"RTP packet length is less than {RTP_HEADER_LENGTH} bytes"
            )

        v_p_x_cc, m_pt, sequence_number,key_first_type,timestamp, ssrc = unpack("!BBHBLL", data[0:13])
        version = v_p_x_cc >> 6
        padding = (v_p_x_cc >> 5) & 1
        extension = (v_p_x_cc >> 4) & 1
        cc = v_p_x_cc & 0x0F
        if version != 2:
            raise ValueError("RTP packet has invalid version")
        if len(data) < RTP_HEADER_LENGTH + 4 * cc:
            raise ValueError("RTP packet has truncated CSRC")

        packet = cls(
            marker=(m_pt >> 7),
            payload_type=(m_pt & 0x7F),
            sequence_number=sequence_number,
            timestamp=timestamp,
            ssrc=ssrc,
        )
        packet.set_is_key_frame(key_first_type>>7)
        packet.set_first_packet_of_frame((key_first_type>>6)&1)
        packet.set_packet_type(key_first_type & 0x3F)
        pos = RTP_HEADER_LENGTH
        for i in range(0, cc):
            packet.csrc.append(unpack_from("!L", data, pos)[0])
            pos += 4

        if extension:
            if len(data) < pos + 4:
                raise ValueError("RTP packet has truncated extension profile / length")
            extension_profile, extension_length = unpack_from("!HH", data, pos)
            extension_length *= 4
            pos += 4

            if len(data) < pos + extension_length:
                raise ValueError("RTP packet has truncated extension value")
            extension_value = data[pos : pos + extension_length]
            pos += extension_length
            packet.extensions = extensions_map.get(extension_profile, extension_value)
            packet.header_size=extension_length
        if padding:
            padding_len = data[-1]
            if not padding_len or padding_len > len(data) - pos:
                raise ValueError("RTP packet padding length is invalid")
            packet.padding_size = padding_len
            packet.payload = data[pos:-padding_len]
        else:
            packet.payload = data[pos:]
        packet.payload_size=len(packet.payload)

        return packet
def unwrap_rtx(rtx: RtpPacketToSend, payload_type: int, ssrc: int) -> RtpPacketToSend:
    """
    Recover initial packet from a retransmission packet.
    """
    packet = RtpPacketToSend(
        payload_type=payload_type,
        marker=rtx.marker,
        sequence_number=unpack("!H", rtx.payload[0:2])[0],
        timestamp=rtx.timestamp,
        ssrc=ssrc,
        payload=rtx.payload[2:],
    )
    packet.csrc = rtx.csrc
    packet.extensions = rtx.extensions
    return packet
def wrap_rtx(
    packet: RtpPacketToSend, payload_type: int, sequence_number: int, ssrc: int
) -> RtpPacketToSend:
    """
    Create a retransmission packet from a lost packet.
    """
    rtx = RtpPacketToSend(
        payload_type=payload_type,
        sequence_number=sequence_number,
        timestamp=packet.timestamp,
        ssrc=ssrc,
        payload=pack("!H", packet.sequence_number) + packet.payload,
    )
    rtx.csrc = packet.csrc
    rtx.extensions = packet.extensions
    return rtx