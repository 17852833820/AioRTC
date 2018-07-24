MAX_MISORDER = 100


class JitterPacket:
    def __init__(self, payload, sequence_number, timestamp):
        self.payload = payload
        self.sequence_number = sequence_number
        self.timestamp = timestamp

    def __repr__(self):
        return 'JitterPacket(seq=%d, ts=%d)' % (self.sequence_number, self.timestamp)


class JitterBuffer:
    def __init__(self, capacity):
        self._capacity = capacity
        self._head = 0
        self._origin = None
        self._packets = [None for i in range(capacity)]

    @property
    def capacity(self):
        return self._capacity

    def add(self, payload, sequence_number, timestamp):
        if self._origin is None:
            self._origin = sequence_number
        elif sequence_number <= self._origin - MAX_MISORDER:
            self.__reset()
            self._origin = sequence_number
        elif sequence_number < self._origin:
            return

        delta = sequence_number - self._origin
        if delta >= 2 * self.capacity:
            # received packet is so far beyond capacity we cannot keep any
            # previous packets, so reset the buffer
            self.__reset()
            self._origin = sequence_number
            delta = 0
        elif delta >= self.capacity:
            # remove just enough packets to fit the received packets
            excess = delta - self.capacity + 1
            self.remove(excess)
            delta = sequence_number - self._origin

        pos = (self._head + delta) % self._capacity
        self._packets[pos] = JitterPacket(payload=payload,
                                          sequence_number=sequence_number,
                                          timestamp=timestamp)

    def peek(self, offset):
        if offset >= self._capacity:
            raise IndexError('Cannot peek at offset %d, capacity is %d' % (offset, self._capacity))
        pos = (self._head + offset) % self._capacity
        return self._packets[pos]

    def remove(self, count):
        assert count <= self._capacity
        packets = [None for i in range(count)]
        for i in range(count):
            packets[i] = self._packets[self._head]
            self._packets[self._head] = None
            self._head = (self._head + 1) % self._capacity
            self._origin += 1
        return packets

    def __reset(self):
        self._head = 0
        self._origin = None

        for i in range(self._capacity):
            self._packets[i] = None
