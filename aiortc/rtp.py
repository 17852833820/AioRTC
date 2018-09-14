import os
from struct import pack, unpack, unpack_from

import attr

# reserved to avoid confusion with RTCP
FORBIDDEN_PAYLOAD_TYPES = range(72, 77)
DYNAMIC_PAYLOAD_TYPES = range(96, 128)

RTP_HEADER_LENGTH = 12
RTCP_HEADER_LENGTH = 8

PACKETS_LOST_MIN = - (1 << 23)
PACKETS_LOST_MAX = (1 << 23) - 1

RTCP_SR = 200
RTCP_RR = 201
RTCP_SDES = 202
RTCP_BYE = 203
RTCP_RTPFB = 205
RTCP_PSFB = 206

RTCP_RTPFB_NACK = 1

RTCP_PSFB_PLI = 1
RTCP_PSFB_SLI = 2
RTCP_PSFB_RPSI = 3
RTCP_PSFB_APP = 15


def clamp_packets_lost(count):
    return max(PACKETS_LOST_MIN, min(count, PACKETS_LOST_MAX))


def pack_packets_lost(count):
    return pack('!l', count)[1:]


def unpack_packets_lost(d):
    if d[0] & 0x80:
        d = b'\xff' + d
    else:
        d = b'\x00' + d
    return unpack('!l', d)[0]


def pack_rtcp_packet(packet_type, count, payload):
    assert len(payload) % 4 == 0
    return pack('!BBH',
                (2 << 6) | count,
                packet_type,
                len(payload) // 4) + payload


def pack_remb_fci(bitrate, ssrcs):
    """
    Pack the FCI for a Receiver Estimated Maximum Bitrate report.

    https://tools.ietf.org/html/draft-alvestrand-rmcat-remb-03
    """
    data = b'REMB'
    exponent = 0
    mantissa = bitrate
    while mantissa > 0x3ffff:
        mantissa >>= 1
        exponent += 1
    data += pack('!BBH',
                 len(ssrcs),
                 (exponent << 2) | (mantissa >> 16),
                 (mantissa & 0xffff))
    for ssrc in ssrcs:
        data += pack('!L', ssrc)
    return data


def unpack_remb_fci(data):
    """
    Unpack the FCI for a Receiver Estimated Maximum Bitrate report.

    https://tools.ietf.org/html/draft-alvestrand-rmcat-remb-03
    """
    if len(data) < 8 or data[0:4] != b'REMB':
        raise ValueError('Invalid REMB prefix')

    exponent = (data[5] & 0xfc) >> 2
    mantissa = ((data[5] & 0x03) << 16) | (data[6] << 8) | data[7]
    bitrate = mantissa << exponent

    pos = 8
    ssrcs = []
    for r in range(data[4]):
        ssrcs.append(unpack_from('!L', data, pos)[0])
        pos += 4

    return (bitrate, ssrcs)


def is_rtcp(msg):
    return len(msg) >= 2 and msg[1] >= 192 and msg[1] <= 208


def padl(l):
    """
    Return amount of padding needed for a 4-byte multiple.
    """
    return 4 * ((l + 3) // 4) - l


def get_header_extensions(packet):
    """
    Parse header extensions according to RFC5285.
    """
    extensions = []
    pos = 0

    if packet.extension_profile == 0xBEDE:
        # One-Byte Header
        while pos < len(packet.extension_value):
            if packet.extension_value[pos] == 0:
                pos += 1
                continue

            x_id = (packet.extension_value[pos] & 0xf0) >> 4
            x_length = (packet.extension_value[pos] & 0x0f) + 1
            pos += 1

            x_value = packet.extension_value[pos:pos + x_length]
            extensions.append((x_id,  x_value))
            pos += x_length
    elif packet.extension_profile == 0x1000:
        # Two-Byte Header
        while pos < len(packet.extension_value):
            if packet.extension_value[pos] == 0:
                pos += 1
                continue

            x_id, x_length = unpack('!BB', packet.extension_value[pos:pos+2])
            pos += 2

            x_value = packet.extension_value[pos:pos + x_length]
            extensions.append((x_id,  x_value))
            pos += x_length

    return extensions


def set_header_extensions(packet, extensions):
    """
    Serialize header extensions according to RFC5285.
    """
    if not extensions:
        packet.extension_profile = 0
        packet.extension_value = None
        return

    one_byte = True
    for x_id, x_value in extensions:
        x_length = len(x_value)
        assert x_id > 0 and x_id < 256
        assert x_length >= 0 and x_length < 256
        if x_id > 14 or x_length == 0 or x_length > 16:
            one_byte = False

    if one_byte:
        # One-Byte Header
        packet.extension_profile = 0xBEDE
        packet.extension_value = b''
        for x_id, x_value in extensions:
            x_length = len(x_value)
            packet.extension_value += pack('!B', (x_id << 4) | (x_length - 1))
            packet.extension_value += x_value
    else:
        # Two-Byte Header
        packet.extension_profile = 0x1000
        packet.extension_value = b''
        for x_id, x_value in extensions:
            x_length = len(x_value)
            packet.extension_value += pack('!BB', x_id, x_length)
            packet.extension_value += x_value

    packet.extension_value += b'\x00' * padl(len(packet.extension_value))


@attr.s
class RtcpReceiverInfo:
    ssrc = attr.ib()
    fraction_lost = attr.ib()
    packets_lost = attr.ib()
    highest_sequence = attr.ib()
    jitter = attr.ib()
    lsr = attr.ib()
    dlsr = attr.ib()

    def __bytes__(self):
        data = pack('!LB', self.ssrc, self.fraction_lost)
        data += pack_packets_lost(self.packets_lost)
        data += pack('!LLLL', self.highest_sequence, self.jitter, self.lsr, self.dlsr)
        return data

    @classmethod
    def parse(cls, data):
        ssrc, fraction_lost = unpack('!LB', data[0:5])
        packets_lost = unpack_packets_lost(data[5:8])
        highest_sequence, jitter, lsr, dlsr = unpack('!LLLL', data[8:])
        return cls(
            ssrc=ssrc,
            fraction_lost=fraction_lost,
            packets_lost=packets_lost,
            highest_sequence=highest_sequence,
            jitter=jitter,
            lsr=lsr,
            dlsr=dlsr
        )


@attr.s
class RtcpSenderInfo:
    ntp_timestamp = attr.ib()
    rtp_timestamp = attr.ib()
    packet_count = attr.ib()
    octet_count = attr.ib()

    def __bytes__(self):
        return pack('!QLLL',
                    self.ntp_timestamp,
                    self.rtp_timestamp,
                    self.packet_count,
                    self.octet_count)

    @classmethod
    def parse(cls, data):
        ntp_timestamp, rtp_timestamp, packet_count, octet_count = unpack('!QLLL', data)
        return cls(
            ntp_timestamp=ntp_timestamp,
            rtp_timestamp=rtp_timestamp,
            packet_count=packet_count,
            octet_count=octet_count)


@attr.s
class RtcpSourceInfo:
    ssrc = attr.ib()
    items = attr.ib()


class RtcpPacket:
    @classmethod
    def parse(cls, data):
        pos = 0
        packets = []

        while pos < len(data):
            if len(data) < RTCP_HEADER_LENGTH:
                raise ValueError('RTCP packet length is less than %d bytes' % RTCP_HEADER_LENGTH)

            v_p_count, packet_type, length = unpack('!BBH', data[pos:pos + 4])
            version = (v_p_count >> 6)
            # padding = ((v_p_count >> 5) & 1)
            count = (v_p_count & 0x1f)
            if version != 2:
                raise ValueError('RTCP packet has invalid version')
            pos += 4
            end = pos + length * 4
            payload = data[pos:end]
            pos = end

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


@attr.s
class RtcpByePacket:
    sources = attr.ib()

    def __bytes__(self):
        payload = b''.join([pack('!L', ssrc) for ssrc in self.sources])
        return pack_rtcp_packet(RTCP_BYE, len(self.sources), payload)

    @classmethod
    def parse(cls, data, count):
        sources = list(unpack('!' + ('L' * count), data))
        return cls(sources=sources)


@attr.s
class RtcpPsfbPacket:
    """"
    Payload-Specific Feedback Message (RFC 4585).
    """
    fmt = attr.ib()
    ssrc = attr.ib()
    media_ssrc = attr.ib()
    fci = attr.ib(default=b'')

    def __bytes__(self):
        payload = pack('!LL', self.ssrc, self.media_ssrc) + self.fci
        return pack_rtcp_packet(RTCP_PSFB, self.fmt, payload)

    @classmethod
    def parse(cls, data, fmt):
        ssrc, media_ssrc = unpack('!LL', data[0:8])
        fci = data[8:]
        return cls(fmt=fmt, ssrc=ssrc, media_ssrc=media_ssrc, fci=fci)


@attr.s
class RtcpRrPacket:
    ssrc = attr.ib()
    reports = attr.ib(default=attr.Factory(list))

    def __bytes__(self):
        payload = pack('!L', self.ssrc)
        for report in self.reports:
            payload += bytes(report)
        return pack_rtcp_packet(RTCP_RR, len(self.reports), payload)

    @classmethod
    def parse(cls, data, count):
        ssrc = unpack('!L', data[0:4])[0]
        pos = 4
        reports = []
        for r in range(count):
            reports.append(RtcpReceiverInfo.parse(data[pos:pos + 24]))
            pos += 24
        return cls(ssrc=ssrc, reports=reports)


@attr.s
class RtcpRtpfbPacket:
    """
    Generic RTP Feedback Message (RFC 4585).
    """
    fmt = attr.ib()
    ssrc = attr.ib()
    media_ssrc = attr.ib()

    # generick NACK
    lost = attr.ib(default=attr.Factory(list))

    def __bytes__(self):
        payload = pack('!LL', self.ssrc, self.media_ssrc)
        if self.lost:
            pid = self.lost[0]
            blp = 0
            for p in self.lost[1:]:
                d = p - pid - 1
                if d < 16:
                    blp |= (1 << d)
                else:
                    payload += pack('!HH', pid, blp)
                    pid = p
                    blp = 0
            payload += pack('!HH', pid, blp)
        return pack_rtcp_packet(RTCP_RTPFB, self.fmt, payload)

    @classmethod
    def parse(cls, data, fmt):
        ssrc, media_ssrc = unpack('!LL', data[0:8])
        lost = []
        for pos in range(8, len(data), 4):
            pid, blp = unpack('!HH', data[pos:pos + 4])
            lost.append(pid)
            for d in range(0, 16):
                if (blp >> d) & 1:
                    lost.append(pid + d + 1)
        return cls(fmt=fmt, ssrc=ssrc, media_ssrc=media_ssrc, lost=lost)


@attr.s
class RtcpSdesPacket:
    chunks = attr.ib(default=attr.Factory(list))

    def __bytes__(self):
        payload = b''
        for chunk in self.chunks:
            payload += pack('!L', chunk.ssrc)
            for d_type, d_value in chunk.items:
                payload += pack('!BB', d_type, len(d_value)) + d_value
            payload += b'\x00\x00'
        while len(payload) % 4:
            payload += b'\x00'
        return pack_rtcp_packet(RTCP_SDES, len(self.chunks), payload)

    @classmethod
    def parse(cls, data, count):
        pos = 0
        chunks = []
        for r in range(count):
            ssrc = unpack('!L', data[pos:pos + 4])[0]
            pos += 4
            items = []
            while True:
                d_type, d_length = unpack('!BB', data[pos:pos + 2])
                pos += 2
                d_value = data[pos:pos + d_length]
                pos += d_length
                if d_type == 0:
                    break
                else:
                    items.append((d_type, d_value))
            chunks.append(RtcpSourceInfo(ssrc=ssrc, items=items))
        return cls(chunks=chunks)


@attr.s
class RtcpSrPacket:
    ssrc = attr.ib()
    sender_info = attr.ib()
    reports = attr.ib(default=attr.Factory(list))

    def __bytes__(self):
        payload = pack('!L', self.ssrc)
        payload += bytes(self.sender_info)
        for report in self.reports:
            payload += bytes(report)
        return pack_rtcp_packet(RTCP_SR, len(self.reports), payload)

    @classmethod
    def parse(cls, data, count):
        ssrc = unpack('!L', data[0:4])[0]
        sender_info = RtcpSenderInfo.parse(data[4:24])
        pos = 24
        reports = []
        for r in range(count):
            reports.append(RtcpReceiverInfo.parse(data[pos:pos + 24]))
            pos += 24
        return RtcpSrPacket(ssrc=ssrc, sender_info=sender_info, reports=reports)


class RtpPacket:
    def __init__(self, payload_type=0, marker=0, sequence_number=0, timestamp=0,
                 ssrc=0, payload=b''):
        self.version = 2
        self.marker = marker
        self.payload_type = payload_type
        self.sequence_number = sequence_number
        self.timestamp = timestamp
        self.ssrc = ssrc
        self.csrc = []
        self.extension_profile = 0
        self.extension_value = None
        self.payload = payload
        self.padding_size = 0

    def __bytes__(self):
        extension = self.extension_value is not None
        padding = self.padding_size > 0
        data = pack(
            '!BBHLL',
            (self.version << 6) | (padding << 5) | (extension << 4) | len(self.csrc),
            (self.marker << 7) | self.payload_type,
            self.sequence_number,
            self.timestamp,
            self.ssrc)
        for csrc in self.csrc:
            data += pack('!L', csrc)
        if self.extension_value is not None:
            data += pack('!HH', self.extension_profile, len(self.extension_value) >> 2)
            data += self.extension_value
        data += self.payload
        if padding:
            data += os.urandom(self.padding_size - 1)
            data += bytes([self.padding_size])
        return data

    def __repr__(self):
        return 'RtpPacket(seq=%d, ts=%s, marker=%d, payload=%d, %d bytes)' % (
            self.sequence_number, self.timestamp, self.marker, self.payload_type, len(self.payload))

    @classmethod
    def parse(cls, data):
        if len(data) < RTP_HEADER_LENGTH:
            raise ValueError('RTP packet length is less than %d bytes' % RTP_HEADER_LENGTH)

        v_p_x_cc, m_pt, sequence_number, timestamp, ssrc = unpack('!BBHLL', data[0:12])
        version = (v_p_x_cc >> 6)
        padding = ((v_p_x_cc >> 5) & 1)
        extension = ((v_p_x_cc >> 4) & 1)
        cc = (v_p_x_cc & 0x0f)
        if version != 2:
            raise ValueError('RTP packet has invalid version')

        packet = cls(
            marker=(m_pt >> 7),
            payload_type=(m_pt & 0x7f),
            sequence_number=sequence_number,
            timestamp=timestamp,
            ssrc=ssrc)

        pos = 12
        for i in range(0, cc):
            packet.csrc.append(unpack('!L', data[pos:pos+4])[0])
            pos += 4

        if extension:
            packet.extension_profile, x_length = unpack('!HH', data[pos:pos+4])
            pos += 4
            packet.extension_value = data[pos:pos+x_length*4]
            pos += x_length * 4

        if padding:
            padding_len = data[-1]
            if not padding_len or padding_len > len(data) - pos:
                raise ValueError('RTP packet padding length is invalid')
            packet.padding_size = padding_len
            packet.payload = data[pos:-padding_len]
        else:
            packet.payload = data[pos:]

        return packet
