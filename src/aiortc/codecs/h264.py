import fractions
import logging
import math
from itertools import tee
from struct import pack, unpack_from
from typing import Iterator, List, Optional, Sequence, Tuple, Type, TypeVar
from enum import Enum

import av
from av.frame import Frame
from av.packet import Packet

from ..jitterbuffer import JitterFrame
from ..mediastreams import VIDEO_TIME_BASE, convert_timebase
from .base import Decoder, Encoder

logger = logging.getLogger(__name__)

DEFAULT_BITRATE = 1000000  # 3000 kbps
MIN_BITRATE = 100000  # 500 kbps
MAX_BITRATE = 3000000  # 3 Mbps

MAX_FRAME_RATE = 30
PACKET_MAX = 1300

NAL_TYPE_FU_A = 28
NAL_TYPE_STAP_A = 24

NAL_HEADER_SIZE = 1
FU_A_HEADER_SIZE = 2
LENGTH_FIELD_SIZE = 2
STAP_A_HEADER_SIZE = NAL_HEADER_SIZE + LENGTH_FIELD_SIZE

DESCRIPTOR_T = TypeVar("DESCRIPTOR_T", bound="H264PayloadDescriptor")
T = TypeVar("T")


 

class NalUnitType(Enum):
    NAL_UNKNOWN = 0
    NAL_SLICE = 1
    NAL_SLICE_DPA = 2
    NAL_SLICE_DPB = 3
    NAL_SLICE_DPC = 4
    NAL_SLICE_IDR = 5
    NAL_SEI = 6
    NAL_SPS = 7
    NAL_PPS = 8


class Frametype(Enum):
    FRAME_I = 0
    FRAME_P = 1
    FRAME_B = 2
    SEI = 3  # 新增 SEI 常量
    SPS = 4  # 新增 SPS 常量
    PPS = 5  # 新增 PPS 常量 
class Bitstream:
    def __init__(self, p_data):
        self.p_start = p_data
        self.p = 0
        self.p_end = self.p + len(p_data)
        self.i_left = 8
    def bs_read(self, i_count):
        i_mask = [0x00, 0x01, 0x03, 0x07, 0x0f, 0x1f, 0x3f, 0x7f,
                 0xff, 0x1ff, 0x3ff, 0x7ff, 0xfff, 0x1fff, 0x3fff,
                 0x7fff, 0xffff, 0x1ffff, 0x3ffff, 0x7ffff, 0xfffff,
                 0x1fffff, 0x3fffff, 0x7fffff, 0xffffff, 0x1ffffff,
                 0x3ffffff, 0x7ffffff, 0xfffffff, 0x1fffffff, 0x3fffffff,
                 0x7fffffff, 0xffffffff]

        i_result = 0

        while i_count > 0:
            if self.p >= self.p_end:
                break

            i_shr = self.i_left - i_count

            if i_shr >= 0:
                i_result |= (self.p >> i_shr) & i_mask[i_count]
                self.i_left -= i_count

                if self.i_left == 0:
                    self.p += 1
                    self.i_left = 8

                return i_result
            else:
                i_result |= (self.p & i_mask[self.i_left]) << -i_shr
                i_count -= self.i_left
                self.p += 1
                self.i_left = 8

        return i_result

    def bs_read1(self):
        if self.p < self.p_end:
            i_result = (self.p >> (self.i_left - 1)) & 0x01
            self.i_left -= 1

            if self.i_left == 0:
                self.p += 1
                self.i_left = 8

            return i_result
        return 0
    def bs_read_ue(s):
        i = 0

        while s.bs_read1() == 0 and s.p < s.p_end and i < 32:
            i += 1

        return (1 << i) - 1 + s.bs_read(i)


def pairwise(iterable: Sequence[T]) -> Iterator[Tuple[T, T]]:
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class H264PayloadDescriptor:
    def __init__(self, first_fragment):
        self.first_fragment = first_fragment

    def __repr__(self):
        return f"H264PayloadDescriptor(FF={self.first_fragment})"

    @classmethod
    def parse(cls: Type[DESCRIPTOR_T], data: bytes) -> Tuple[DESCRIPTOR_T, bytes]:
        output = bytes()

        # NAL unit header
        if len(data) < 2:
            raise ValueError("NAL unit is too short")
        nal_type = data[0] & 0x1F
        f_nri = data[0] & (0x80 | 0x60)
        pos = NAL_HEADER_SIZE

        if nal_type in range(1, 24):
            # single NAL unit
            output = bytes([0, 0, 0, 1]) + data
            obj = cls(first_fragment=True)
        elif nal_type == NAL_TYPE_FU_A:
            # fragmentation unit
            original_nal_type = data[pos] & 0x1F
            first_fragment = bool(data[pos] & 0x80)
            pos += 1

            if first_fragment:
                original_nal_header = bytes([f_nri | original_nal_type])
                output += bytes([0, 0, 0, 1])
                output += original_nal_header
            output += data[pos:]

            obj = cls(first_fragment=first_fragment)
        elif nal_type == NAL_TYPE_STAP_A:
            # single time aggregation packet
            offsets = []
            while pos < len(data):
                if len(data) < pos + LENGTH_FIELD_SIZE:
                    raise ValueError("STAP-A length field is truncated")
                nalu_size = unpack_from("!H", data, pos)[0]
                pos += LENGTH_FIELD_SIZE
                offsets.append(pos)

                pos += nalu_size
                if len(data) < pos:
                    raise ValueError("STAP-A data is truncated")

            offsets.append(len(data) + LENGTH_FIELD_SIZE)
            for start, end in pairwise(offsets):
                end -= LENGTH_FIELD_SIZE
                output += bytes([0, 0, 0, 1])
                output += data[start:end]

            obj = cls(first_fragment=True)
        else:
            raise ValueError(f"NAL unit type {nal_type} is not supported")

        return obj, output


class H264Decoder(Decoder):
    def __init__(self) -> None:
        self.codec = av.CodecContext.create("h264", "r")

    def decode(self, encoded_frame: JitterFrame) -> List[Frame]:
        try:
            packet = av.Packet(encoded_frame.data)
            packet.pts = encoded_frame.timestamp
            packet.time_base = VIDEO_TIME_BASE
            frames = self.codec.decode(packet)
        except av.AVError as e:
            logger.warning(
                "H264Decoder() failed to decode, skipping package: " + str(e)
            )
            return []

        return frames

"""初始化编码器"""

       
def create_encoder_context(
    codec_name: str, width: int, height: int, bitrate: int
) -> Tuple[av.CodecContext, bool]:
    
    # av.VideoCodecContext.max_b_frames = property(lambda self: self.ptr.max_b_frames)
    
    codec = av.CodecContext.create(codec_name, "w")
   
    codec.width = width
    codec.height = height
    codec.bit_rate = bitrate
    codec.pix_fmt = "yuv420p"
    codec.framerate = fractions.Fraction(MAX_FRAME_RATE, 1)
    codec.time_base = fractions.Fraction(1, MAX_FRAME_RATE)
    codec.options = {
        "profile": "main",# baseline, main, high, high10, high422, high444.
        "level": "31",
        "tune": "zerolatency",  # does nothing using h264_omx
    }
    codec.gop_size = 99999  # GOP (Group of Pictures) 大小
    if "crf" in codec.options:
        del codec.options["crf"]
    # codec.qmin = 30  # 最小量化器
    # codec.qmax = 51  # 最大量化器
    # codec.has_b_frames =True
    # codec.max_b_frames = 1  # 最大 B 帧数
    # codec.global_quality = 10  # 全局质量
    codec.open()
    return codec, codec_name == "h264_omx"


class H264Encoder(Encoder):
    def __init__(self) -> None:
        self.buffer_data = b""
        self.buffer_pts: Optional[int] = None
        self.codec: Optional[av.CodecContext] = None
        self.codec_buffering = False
        self.__target_bitrate = DEFAULT_BITRATE
        self.frame_size=0

    @staticmethod
    def _packetize_fu_a(data: bytes) -> List[bytes]:
        available_size = PACKET_MAX - FU_A_HEADER_SIZE
        payload_size = len(data) - NAL_HEADER_SIZE
        num_packets = math.ceil(payload_size / available_size)
        num_larger_packets = payload_size % num_packets
        package_size = payload_size // num_packets

        f_nri = data[0] & (0x80 | 0x60)  # fni of original header
        nal = data[0] & 0x1F

        fu_indicator = f_nri | NAL_TYPE_FU_A

        fu_header_end = bytes([fu_indicator, nal | 0x40])
        fu_header_middle = bytes([fu_indicator, nal])
        fu_header_start = bytes([fu_indicator, nal | 0x80])
        fu_header = fu_header_start

        packages = []
        offset = NAL_HEADER_SIZE
        while offset < len(data):
            if num_larger_packets > 0:
                num_larger_packets -= 1
                payload = data[offset : offset + package_size + 1]
                offset += package_size + 1
            else:
                payload = data[offset : offset + package_size]
                offset += package_size

            if offset == len(data):
                fu_header = fu_header_end

            packages.append(fu_header + payload)

            fu_header = fu_header_middle
        assert offset == len(data), "incorrect fragment data"

        return packages

    @staticmethod # 
    def _packetize_stap_a(
        data: bytes, packages_iterator: Iterator[bytes]
    ) -> Tuple[bytes, bytes]:
        counter = 0
        available_size = PACKET_MAX - STAP_A_HEADER_SIZE

        stap_header = NAL_TYPE_STAP_A | (data[0] & 0xE0)

        payload = bytes()
        try:
            nalu = data  # with header
            while len(nalu) <= available_size and counter < 9:
                stap_header |= nalu[0] & 0x80

                nri = nalu[0] & 0x60
                if stap_header & 0x60 < nri:
                    stap_header = stap_header & 0x9F | nri

                available_size -= LENGTH_FIELD_SIZE + len(nalu)
                counter += 1
                payload += pack("!H", len(nalu)) + nalu
                nalu = next(packages_iterator)

            if counter == 0:
                nalu = next(packages_iterator)
        except StopIteration:
            nalu = None

        if counter <= 1:
            return data, nalu
        else:
            return bytes([stap_header]) + payload, nalu
    """用于拆分 H.264 编码的比特流（bitstream）中的不同 NAL 单元"""
    @staticmethod
    def _split_bitstream(buf: bytes) -> Iterator[bytes]:
        # Translated from: https://github.com/aizvorski/h264bitstream/blob/master/h264_nal.c#L134
        i = 0
        while True:
            # Find the start of the NAL unit.
            #
            # NAL Units start with the 3-byte start code 0x000001 or
            # the 4-byte start code 0x00000001.  # 查找 NAL 单元的开始码
            i = buf.find(b"\x00\x00\x01", i)
            if i == -1: # 如果没有找到，结束函数
                return

            # Jump past the start code
            i += 3
            nal_start = i

            # Find the end of the NAL unit (end of buffer OR next start code) # 查找 NAL 单元的结束码（下一个开始码或数据末尾）
            i = buf.find(b"\x00\x00\x01", i)
            if i == -1:# 如果没有找到下一个开始码，返回当前开始码到数据末尾的部分
                yield buf[nal_start : len(buf)]
                return
            elif buf[i - 1] == 0:
                # 4-byte start code case, jump back one byte
                yield buf[nal_start : i - 1]# 如果是 4 字节的开始码，回退一字节
            else:
                yield buf[nal_start:i] # 返回当前开始码到下一个开始码之前的部分

    @classmethod #用于将给定的数据包（每个经过H264编码后的NAL单元）进行分片
    def _packetize(cls, packages: Iterator[bytes]) -> Tuple[List[bytes],bytes]:
        packetized_packages = []

        packages_iterator = iter(packages)
        package = next(packages_iterator, None)#获取生成器中下一个编码后的NAL单元数据
        package_nal=package
        while package is not None:
            if len(package) > PACKET_MAX:#如果超过了一个Packet，用_packetize_fu_a将NAL单元分成多个FU-A
                packetized_packages.extend(cls._packetize_fu_a(package))
                package = next(packages_iterator, None)
            else:#否则用_packetize_stap_a方法将多个小的NAL单元打包成一个STAP-A
                packetized, package = cls._packetize_stap_a(package, packages_iterator)
                packetized_packages.append(packetized)

        return packetized_packages,package_nal #Packet列表
    """生成器"""
    def _encode_frame(
        self, frame: av.VideoFrame, force_keyframe: bool
    ) -> Iterator[bytes]:
        """检查帧尺寸和比特率是否与先前的编码器匹配：如果尺寸不一致或比特率变化超过10%，重新初始化编码器"""
        if self.codec and (
            frame.width != self.codec.width
            or frame.height != self.codec.height
            # we only adjust bitrate if it changes by over 10%
            or abs(self.target_bitrate - self.codec.bit_rate) / self.codec.bit_rate
            > 0.1
        ):
            self.buffer_data = b""
            self.buffer_pts = None
            self.codec = None
        """是否需要强制生成关键帧"""
        if force_keyframe:
            # force a complete image
            frame.pict_type = av.video.frame.PictureType.I
        else:
            # reset the picture type, otherwise no B-frames are produced
            frame.pict_type = av.video.frame.PictureType.NONE
        """如果编码器未初始化：初始化编码器"""
        if self.codec is None:
            try:
                self.codec, self.codec_buffering = create_encoder_context(
                    "libx264",
                    frame.width,
                    frame.height,
                    bitrate=self.target_bitrate,
                )
                logger.info("Encodec | Frame height: {0} ,width: {1}".format(frame.height,frame.width))
            except Exception:
                logger.error("libx264 error")
        """循环编码"""
        data_to_send = b""
        for package in self.codec.encode(frame):
            package_bytes = bytes(package)
            if self.codec_buffering:#如果有buffer缓冲区：延迟发送以确保累计所有给定PTS的数据
                # delay sending to ensure we accumulate all packages
                # for a given PTS
                if package.pts == self.buffer_pts:
                    self.buffer_data += package_bytes
                else:
                    data_to_send += self.buffer_data
                    self.buffer_data = package_bytes
                    self.buffer_pts = package.pts
            else:
                data_to_send += package_bytes
        self.frame_size=len(data_to_send)
        # logger.info("Encodec | Frame  Type: {0} ,size: {1}".format(frame.pict_type,len(data_to_send)))
        # logger.info("Encodec | Actual encode_bitrate: {0}".format(self.target_bitrate))

        if data_to_send: #将累积的编码数据分割为较小的数据包，并通过_split_bitstream方法发送
            yield from self._split_bitstream(data_to_send)#将 data_to_send 中经过 _split_bitstream 处理后的每个 NAL 单元的内容逐一返回给调用方
    def get_frame_type(self,package_nal)->Frametype:
        # 获取NAL单元的类型
        cNalu =hex(package_nal[0])
        nal_unit_type = int(hex((package_nal[0]) & 0x1F),16)
        s=Bitstream(package_nal)
        if nal_unit_type == NalUnitType.NAL_SLICE.value or nal_unit_type == NalUnitType.NAL_SLICE_IDR.value:
            s.bs_read_ue()  # i_first_mb
            frame_type = s.bs_read_ue()  # picture type
            if frame_type in [0, 5]:
                frametype = Frametype.FRAME_P
            elif frame_type in [1, 6]:
                frametype = Frametype.FRAME_B
            elif frame_type in [3, 8]:
                frametype = Frametype.FRAME_P
            elif frame_type in [2, 7]:
                frametype = Frametype.FRAME_I
            elif frame_type in [4, 9]:
                frametype = Frametype.FRAME_I
                I_Frame_Num += 1
        elif nal_unit_type == NalUnitType.NAL_SEI.value:
            frametype = Frametype.SEI
        elif nal_unit_type == NalUnitType.NAL_SPS.value:
            frametype = Frametype.SPS
        elif nal_unit_type == NalUnitType.NAL_PPS.value:
            frametype = Frametype.PPS
        # logger.info("Encodec | Nal_unit_type: {0} Frame type ==============={1} ".format(nal_unit_type,frametype.name))
        return frametype
    def encode(
        self, frame: Frame, force_keyframe: bool = False
    ) -> Tuple[List[bytes], int]:
        assert isinstance(frame, av.VideoFrame)
        packages = self._encode_frame(frame, force_keyframe) #返回生成器
        timestamp = convert_timebase(frame.pts, frame.time_base, VIDEO_TIME_BASE)
        packages_packetize,package_nal=self._packetize(packages)
        frametype=self.get_frame_type(package_nal)
        
        return packages_packetize, timestamp ,frametype,self.frame_size#返回编码打包后的packet列表和时间戳

    def pack(self, packet: Packet) -> Tuple[List[bytes], int]:
        assert isinstance(packet, av.Packet)
        packages = self._split_bitstream(bytes(packet))
        timestamp = convert_timebase(packet.pts, packet.time_base, VIDEO_TIME_BASE)
        return self._packetize(packages), timestamp

    @property
    def target_bitrate(self) -> int:
        """
        Target bitrate in bits per second.
        """
        return self.__target_bitrate
    # 如何调节目标比特率
    @target_bitrate.setter
    def target_bitrate(self, bitrate: int) -> None:
        bitrate = max(MIN_BITRATE, min(bitrate, MAX_BITRATE))
        self.__target_bitrate = bitrate


def h264_depayload(payload: bytes) -> bytes:
    descriptor, data = H264PayloadDescriptor.parse(payload)
    return data
