AppRTC client
=============

This example illustrates how to connect to Google's AppRTC demo application.
It also illustrates:

- how to use a MediaPlayer to read media from a file

- how to use a MediaRecorder to capture media to a file

First install the required packages:

.. code-block:: console

    $ pip install aiohttp aiortc opencv-python websockets

When you run the example, it will connect to AppRTC and wait for a participant
to join the room:

.. code-block:: console

   $ python apprtc.py

You will be given a URL which you can point your browser to in order to join
the room.

Credits
-------

Photo by Xiao jinshi on Unsplash.
