import React, { Component } from 'react';
import ReactQuill from 'react-quill';
import 'react-quill/dist/quill.snow.css';
// import 'react-quill/dist/quill.bubble.css';
import 'react-quill/dist/quill.core.css';
import QuillEditor from './editor.style';

export default class Editor extends Component {
  state = { value: this.props.value || '' };
  quillModules = {
    toolbar: {
      container: [
        [{ header: [1, 2, false] }, { font: [] }],
        ['bold', 'italic', 'underline', 'strike', 'blockquote'],
        [
          { list: 'ordered' },
          { list: 'bullet' },
          { indent: '-1' },
          { indent: '+1' },
        ],
        // ['link', 'image', 'video'],
        [],
        ['clean'],
      ],
    },
  };
  handleChange = value => {
    this.setState({ value });
    if (this.props.onChange) {
      this.props.onChange(value);
    }
  };
  componentWillReceiveProps(nextProps) {
    if (nextProps.value !== this.state.value)
      this.setState({ value: nextProps.value });
  }
  render() {
    const options = {
      theme: 'snow',
      formats: Editor.formats,
      placeholder: this.props.placeholder || 'Write Something',
      value: this.state.value,
      onChange: this.handleChange,
      modules: this.quillModules,
    };
    return (
      <QuillEditor className={this.props.className}>
        <ReactQuill {...options} />
      </QuillEditor>
    );
  }
}
