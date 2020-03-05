import styled from 'styled-components';

const CommentsWrapper = styled.div`
  border-top: 1px solid #e2e2e2;
  margin-top: 30px;

  .comment-inner {
    padding: 30px 0 16px;
    display: flex;
  }

  .comment.admin {
    .comment-inner {
      .comment-content {
        width: 100%;
      }
    }
  }

  .comment-avatar {
    flex-shrink: 0;
    position: relative;
    margin-right: 16px;
    cursor: pointer;
    width: 40px;
    height: 40px;
    border-radius: 50%;
    overflow: hidden;
    img {
      width: 100%;
      height: 100%;
      object-fit: cover;
    }
  }

  .comment-author {
    margin-bottom: 4px;
    font-size: 14px;
    display: flex;
    justify-content: flex-start;
  }

  .comment-author > a,
  .comment-author > span {
    height: auto;
    padding-right: 8px;
    font-size: 16px;
    line-height: 30px;
    color: #788195;
    font-family: 'Roboto';
    font-weight: 500;
  }

  .author-time {
    font-size: 14px;
    line-height: 30px;
    color: #8c90b5;
    font-family: 'Roboto';
    font-weight: 400;
  }

  .comment-detail {
    font-size: 14px;
    line-height: 22px;
    color: #797979;
    font-family: 'Roboto';
    font-weight: 400;
  }

  .comment-action {
    span {
      color: rgba(0, 0, 0, 0.45);
      font-size: 12px;
      cursor: pointer;
      &:hover {
        color: rgba(0, 0, 0, 0.85);
      }
    }
  }

  .comment-form {
    display: flex;
    align-items: flex-start;
    flex-direction: column;
    .MuiFormControl-root {
      margin-top: 0;
      margin-bottom: 30px;
      width: 100%;
    }
  }
`;

export default CommentsWrapper;
