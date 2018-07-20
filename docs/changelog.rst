Changelog
=========

0.9.1
-----

Data channels
.............

  * Revert making RTCDataChannel.send a coroutine.

0.9.0
-----

Media
.....

  * Enable post-processing in VP8 decoder to remove (macro) blocks.

  * Set target bitrate for VP8 encoder to 900kbps.

  * Re-create VP8 encoder if frame size changes.

  * Implement jitter estimation for RTCP reports.

  * Avoid overflowing the DLSR field for RTCP reports.

  * Raise video jitter buffer size.

Data channels
.............

  * BREAKING CHANGE: make RTCDataChannel.send a coroutine.

  * Support spec-compliant SDP format for datachannels, as used in Firefox 63.

  * Never send a negative advertised_cwnd.

Examples
........

  * `datachannel-filexfer`: new example for file transfer over a data channel.

  * `datachannel-vpn`: new example for a VPN over a data channel.

  * `server`: make it possible to select video resolution.

0.8.0
-----

Media
.....

  * Align VP8 settings with those used by WebRTC project, which greatly improves
    video quality.

  * Send RTCP source description, sender report, receiver report and bye packets.

Examples
........

  * `server`:

    - make it possible to not transform video at all.

    - allow video display to be up to 1280px wide.

  * `videostream-cli`:

    - fix Python 3.5 compatibility

Miscellaneous
.............

  * Delay logging string interpolation to reduce cost of packet logging in
    non-verbose mode.

0.7.0
-----

Peer connection
...............

  * Add addIceCandidate() method to :class:`aiortc.RTCPeerConnection` to handle
    trickled ICE candidates.

Media
.....

  * Make stop() methods of :class:`aiortc.RTCRtpReceiver`, :class:`aiortc.RTCRtpSender`
    and :class:`RTCRtpTransceiver` coroutines to enable clean shutdown.

Data channels
.............

  * Clean up :class:`aiortc.RTCDataChannel` shutdown sequence.

  * Support receiving an SCTP `RE-CONFIG` to raise number of inbound streams.

Examples
........

  * `server`:

    - perform some image processing using OpenCV.

    - make it possible to disable data channels.

    - make demo web interface more mobile-friendly.

  * `apprtc`:

    - automatically create a room if no room is specified on command line.

    - handle `bye` command.

0.6.0
-----

Peer connection
...............

  * Make it possible to specify one STUN server and / or one TURN server.

  * Add `BUNDLE` support to use a single ICE/DTLS transport for multiple media.

  * Move media encoding / decoding off the main thread.

Data channels
.............

  * Use SCTP `ABORT` instead of `SHUTDOWN` when stopping :class:`aiortc.RTCSctpTransport`.

  * Advertise support for SCTP `RE-CONFIG` extension.

  * Make :class:`aiortc.RTCDataChannel` emit `open` and `close` events.

Examples
........

  * Add an example of how to connect to appr.tc.

  * Capture audio frames to a WAV file in server example.

  * Show datachannel open / close events in server example.
