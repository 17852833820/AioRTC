import asyncio
import datetime

import aioice


def get_ntp_seconds():
    return int((
        datetime.datetime.utcnow() - datetime.datetime(1900, 1, 1, 0, 0, 0)
    ).total_seconds())


class RTCPeerConnection:
    def __init__(self):
        self.__iceConnection = None
        self.__iceGatheringState = 'new'

    @property
    def iceGatheringState(self):
        return self.__iceGatheringState

    async def createAnswer(self):
        return {
            'sdp': self.__createSdp(),
            'type': 'answer',
        }

    async def createOffer(self):
        self.__iceConnection = aioice.Connection(ice_controlling=True)
        self.__iceGatheringState = 'gathering'
        await self.__iceConnection.gather_candidates()
        self.__iceGatheringState = 'complete'

        return {
            'sdp': self.__createSdp(),
            'type': 'offer',
        }

    async def setLocalDescription(self, sessionDescription):
        pass

    async def setRemoteDescription(self, sessionDescription):
        if self.__iceConnection is None:
            self.__iceConnection = aioice.Connection(ice_controlling=False)
            self.__iceGatheringState = 'gathering'
            await self.__iceConnection.gather_candidates()
            self.__iceGatheringState = 'complete'

        for line in sessionDescription['sdp'].splitlines():
            if line.startswith('a=') and ':' in line:
                attr, value = line[2:].split(':', 1)
                if attr == 'candidate':
                    self.__iceConnection.remote_candidates.append(aioice.Candidate.from_sdp(value))
                elif attr == 'ice-ufrag':
                    self.__iceConnection.remote_username = value
                elif attr == 'ice-pwd':
                    self.__iceConnection.remote_password = value
        asyncio.ensure_future(self.__iceConnection.connect())

    def __createSdp(self):
        ntp_seconds = get_ntp_seconds()
        sdp = [
            'v=0',
            'o=- %d %d IN IP4 0.0.0.0' % (ntp_seconds, ntp_seconds),
            's=-',
            't=0 0',
        ]

        sdp += [
            'c=IN IP4 0.0.0.0',
        ]
        for candidate in self.__iceConnection.local_candidates:
            sdp += ['a=candidate:%s' % candidate.to_sdp()]
        sdp += [
            'a=ice-pwd:%s' % self.__iceConnection.local_password,
            'a=ice-ufrag:%s' % self.__iceConnection.local_username,
        ]
        return '\r\n'.join(sdp) + '\r\n'
