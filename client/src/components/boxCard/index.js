import React, { Component } from 'react';
import { BoxCard } from './style';

export default class BoxCards extends Component {
  render() {
    const { title, img, date, message } = this.props;

    return (
      <BoxCard>
        {img ? (
          <div className="imgWrapper">
            <img alt="#" src={`${img}`} />
          </div>
        ) : (
          ''
        )}

        <div className="listContent">
          <div className="listHead">
            {title ? <h5>{title}</h5> : ''}
            {date ? <span className="date">{date}</span> : ''}
          </div>
          {message ? <p>{message}</p> : ''}
        </div>
      </BoxCard>
    );
  }
}
