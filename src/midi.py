from midiutil.MidiFile import MIDIFile


class MIDIMaker:
    def __init__(self):
        self.file = 0
        self.notes = ""
        self.line = ""
        self._setup_midi()

    def from_file(self, file_name: str):
        self.file = open(file_name, "r")
        self.line = self.file.readline()
        counter = 0
        while self.line != "":
            for i, bit in enumerate(self.line):
                if bit == '1':
                    time = (counter * (len(self.line) - 1) + i) / 4
                    self.add_note(self, time)
            self.line = self.file.readline()
            counter += 1
        self.file.close()
        self.write_midi()

    def _setup_midi(self, number_of_tracks=1, track=0, tempo=146):
        # create a MIDI object
        self.midi = MIDIFile(number_of_tracks)

        # initialise the MIDI object
        time = 0
        self.midi.addTrackName(track, time, "Drum Beat")
        self.midi.addTempo(track, time, tempo)

    @staticmethod
    def add_note(self, time, pitch=60, duration=1, channel=9, volume=100, track=0):
        self.midi.addNote(track, channel, pitch, time, duration, volume)

    def write_midi(self):
        # write it to disk
        with open("../output.mid", 'wb') as out:
            self.midi.writeFile(out)


if __name__ == "__main__":
    M = MIDIMaker()
    M.from_file("data.txt")
