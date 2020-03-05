import React from 'react';
import moment from 'moment';
import Button from '@material-ui/core/Button';
import TextField from '@material-ui/core/TextField';
import Tooltip from '@material-ui/core/Tooltip';
import CommentsWrapper from './Comments.style';

const Editor = ({ onChange, onSubmit, value }) => (
  <form className="comment-form" onSubmit={onSubmit}>
    <TextField
      id="comment-message"
      label="Write a comment"
      value={value}
      onChange={onChange}
      margin="normal"
    />
    <Button elementtype="submit" variant="contained" color="primary">
      Add Comment
    </Button>
  </form>
);

class Comments extends React.Component {
  state = {
    comments: [
      {
        actions: [<span>Reply to</span>],
        author: 'Han Solo',
        avatar: 'https://randomuser.me/api/portraits/men/1.jpg',
        content: (
          <p>
            We supply a series of design principles, practical patterns and high
            quality design resources (Sketch and Axure), to help people create
            their product prototypes beautifully and efficiently.
          </p>
        ),
        datetime: (
          <Tooltip
            title={moment()
              .subtract(1, 'days')
              .format('YYYY-MM-DD HH:mm:ss')}
          >
            <span>
              {moment()
                .subtract(1, 'days')
                .fromNow()}
            </span>
          </Tooltip>
        ),
      },
    ],
    submitting: false,
    value: '',
  };

  handleSubmit = event => {
    event.preventDefault();
    if (!this.state.value) {
      return;
    }

    this.setState({
      submitting: true,
    });

    setTimeout(() => {
      this.setState({
        submitting: false,
        value: '',
        comments: [
          {
            actions: [<span>Reply to</span>],
            author: 'Han Solo',
            avatar: 'https://randomuser.me/api/portraits/men/1.jpg',
            content: <p>{this.state.value}</p>,
            datetime: moment().fromNow(),
          },
          ...this.state.comments,
        ],
      });
    }, 1000);
  };

  handleChange = e => {
    this.setState({
      value: e.target.value,
    });
  };

  render() {
    const { comments, submitting, value } = this.state;

    return (
      <CommentsWrapper>
        {comments.length > 0 &&
          comments.map((item, index) => (
            <div key={`comment-key${index}`} className="comment">
              <div className="comment-inner">
                <div className="comment-avatar">
                  <img
                    src="https://randomuser.me/api/portraits/men/1.jpg"
                    alt="Han Solo"
                  />
                </div>
                <div className="comment-content">
                  <div className="comment-author">
                    <span className="author-name">{item.author}</span>
                    <span className="author-time">{item.datetime}</span>
                  </div>
                  <div className="comment-detail">{item.content}</div>
                  <div className="comment-action">
                    <span>Reply to</span>
                  </div>
                </div>
              </div>
            </div>
          ))}

        <div className="comment admin">
          <div className="comment-inner">
            <div className="comment-avatar">
              <img
                src="https://randomuser.me/api/portraits/men/1.jpg"
                alt="Han Solo"
              />
            </div>
            <div className="comment-content">
              <Editor
                onChange={this.handleChange}
                onSubmit={this.handleSubmit}
                submitting={submitting}
                value={value}
              />
            </div>
          </div>
        </div>
      </CommentsWrapper>
    );
  }
}

export default Comments;
